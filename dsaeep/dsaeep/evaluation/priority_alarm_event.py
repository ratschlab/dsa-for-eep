# ===========================================
#
# Event Eval. Code for Priority/Alarm Func. Idea
#
# ===========================================
import logging
import pickle
import time
from functools import partial
from pathlib import Path

import gin
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsaeep.data.hirid.constants import (
    BATCH_PARQUET_PATTERN,
    DATETIME,
    PID,
    REL_DATETIME,
)
from dsaeep.data.utils import q_exp_param
from dsaeep.evaluation.event import (
    EventEvaluation,
    compute_alarms,
    compute_alarms_at_thresholds,
    compute_endevent_array,
    compute_onset_array,
    event_merge,
    mask_reset_time_steps_to_unlabeled,
)
from dsaeep.pipeline import PipelineState, StatefulPipelineStage
from dsaeep.train.sequence import SurvivalWrapper, TabularWrapper
from dsaeep.train.utils import gin_config_to_readable_dictionary


# ----------------------------------------------------------------------------------------
#
# JiT Compiled Evaluation Functions
#
# ----------------------------------------------------------------------------------------
@numba.njit
def base_silencing_function(
    pred_arr: np.ndarray,
    tau_thresh: float,
    current_index: int,
    last_alarm_index: int,
    last_alarm_value: int,
    silence_time_steps: int,
) -> (np.ndarray, np.ndarray):
    """
    Base silencing function that raise an alarm if :
    - score above threshold and not currently silenced

    """

    # check if we're silenced
    # - there has been an alarm in the past
    # - the last alarm is within the window of silencing
    if last_alarm_index != -1 and current_index <= last_alarm_index + silence_time_steps:
        return False, 0

    # we are not silenced, check if we raise an alarm

    # get thresholded binarized alarms
    pot_alarm_arr = pred_arr >= tau_thresh

    # pool the alarms
    alarm = np.sum(pot_alarm_arr) > 0

    return alarm, pred_arr.max()

@numba.njit
def scaling_priority_func(pred_scores: np.ndarray, weights: np.ndarray) -> np.ndarray:

    # Compute the logistic regression
    proba = pred_scores * weights

    return proba


@numba.njit
def compute_alarms_priority(
    inevent_arr: np.ndarray,
    onset_arr: np.ndarray,
    next_event_indeces: np.ndarray,
    next_endevent_indeces: np.ndarray,
    pred_arr: np.ndarray,
    labeled_point_arr: np.ndarray,
    tau_threshs: np.ndarray,
    silence_time_steps: int = 30 // 5,
    reset_time_steps: int = 25 // 5,
    window_time_steps: int = 480 // 5,
    one_true_per_event: bool = False,
    detect_full_past_window: bool = False,
    silencing_function: callable = base_silencing_function,
):
    """
    inevent_arr: np.ndarray
        bool array if we are inside an event
    onset_arr: np.ndarray
        bool array to mark beginning of events
    next_event_indeces: np.ndarray
        array of the next event beginning index for each timepoint
    next_endevent_indeces: np.ndarray
        array of the next event end index for each timepoint
    pred_arr: np.ndarray [seq_len, num_horizons]
        array of prediction scores
    labeled_point_arr: np.ndarray
        bool array to mark labeled points
    tau_thresh: float
        threshold for the prediction scores
    silence_time_steps: int
        number of time steps to silence the alarm
    reset_time_steps: int
        number of time steps to reset the alarm after an event
    window_time_steps: int
        number of time steps to consider for the alarm window / the horizon
    one_true_per_event: bool
        whether to only consider one true alarm per event i.e. only the first alarm
        will be a positive true alarm, subsequent ones for the same event
        are neither false nor true
    detect_full_past_window: bool
        In the CircEWS paper (caption Figure 2) and also here: https://github.com/ratschlab/circEWS/blob/b2b1f00dac4f5d46856a2c7abe2ca4f12d4c612d/evaluation/precision_recall.py#LL328C6-L328C6
        it is stated that an event is considered caught if in the full 8h window before it there has been an alarm.
        Per default we do not consider this and only consider either the full 8h window or (if shorter) only the window
        to the last event end.

    Compute alarms, true alarms and caught events
    given a threshold and a silencing period
    """
    seq_len = inevent_arr.shape[0]
    out_of_bounds_index = seq_len + 1

    silenced_reset_loc_arr = np.zeros(seq_len, dtype=numba.types.bool_)

    raised_alarm_arr = np.zeros(seq_len, dtype=numba.types.bool_)
    true_alarm_arr = np.zeros(seq_len, dtype=numba.types.bool_)
    detected_events = np.zeros(seq_len, dtype=numba.types.bool_)
    detected_distance = np.zeros(seq_len, dtype=numba.types.int64)
    first_alarm_detected_distance = np.zeros(seq_len, dtype=numba.types.int64)
    raised_alarm_count = 0
    true_alarm_count = 0

    last_alarm_index = -1
    last_alarm_value = tau_threshs
    last_endevent_index = 0

    # the last timepoint does not need to be processed
    # as we do not raise an alarm if the stay is over
    for i in range(seq_len - 1):

        # we are at an event boundary, perform the reset
        # this includes the right boundary in the reset (e.g. for 20min reset, we mask 4 timepoints after the event)
        if i == next_endevent_indeces[i]:
            silenced_reset_loc_arr[i + 1 : i + reset_time_steps + 1] = True
            last_endevent_index = i

            # ATTENTION: reset the last alarm, as we enter a event, after it
            # a new episode starts and we do not want to count the last alarm
            # because we used it to detect silencing periods
            last_alarm_index = -1
            last_alarm_value = tau_threshs

        # check detected events
        # - if we are at the beginning of an event
        # - there has been an alarm in the past
        # - the last alarm is within the window
        # - the last alarm happened afte end of last event or we consider full past window
        if (
            onset_arr[i]
            and last_alarm_index != -1
            and last_alarm_index + window_time_steps >= i
            and (last_alarm_index > last_endevent_index or detect_full_past_window)
        ):
            detected_events[i] = True
            detected_distance[i] = max(i - last_alarm_index, detected_distance[i])

        # ATTENTION: reset the last alarm, as we enter a event, after it
        # a new episode starts and we do not want to count the last alarm
        # because we use it to detect silencing periods in `silencing_function`
        if i == next_endevent_indeces[i]:
            last_alarm_index = -1
            last_alarm_value = tau_threshs

        # we step through because:
        # - this is anyway not an alarm location
        # - the location is silenced
        # - the location is not a labeled point
        # if not pot_alarm_arr[i] or silenced_reset_loc_arr[i] or not labeled_point_arr[i]:
        #     continue

        # we handle alarm location and silencing in the `silencing_function`,
        # here we handle purely labeled and reset locations
        if not labeled_point_arr[i] or silenced_reset_loc_arr[i]:
            continue

        # this is a non-silenced pot. alarm location

        # we are inside an event, we step through
        # - current step is before an end
        # - current step is an event beginning or the next start is after the next end
        if i <= next_endevent_indeces[i] and (
            onset_arr[i] or next_endevent_indeces[i] < next_event_indeces[i]
        ):
            # e.g. 118 <= 160 and 160 < 118
            continue

        # =============== PACK INTO A SINGLE ALARM FUNCTION ===============
        # raise alarm and silence
        # prev_last_alarm_index = last_alarm_index

        # check if we raise an alarm
        raised_alarm, raised_value = silencing_function(
            pred_arr[i, :],
            tau_threshs,
            i,
            last_alarm_index,
            last_alarm_value,
            silence_time_steps,
        )

        # we do not raise an alarm
        if not raised_alarm:
            continue

        # we raise an alarm at this location
        last_alarm_index = i
        last_alarm_value = raised_value
        raised_alarm_arr[i] = True
        raised_alarm_count += 1

        # silence the alarm for silencing period
        # or until the next event
        # because as we enter an event the alarm is being reset
        # this includes the right boundary in the silencing (e.g. for 25min silencing, we mask 5 timepoints after the alarm)
        # silence_end_index = min(i + silence_time_steps + 1, next_event_indeces[i] + 1)  # Silencing now handled in silencing function
        # silenced_reset_loc_arr[i + 1 : silence_end_index] = True # Silencing now handled in silencing function
        # =============== PACK INTO A SINGLE ALARM FUNCTION ===============

        # check if the alarm is a True Alarm
        # - if the next event index is out of bounds, there will be no more event
        # - if the next event index is within the window, the alarm is a true alarm
        next_event_index = next_event_indeces[i]
        if next_event_index != out_of_bounds_index and i + window_time_steps >= next_event_index:

            # update first_alarm_detected_distance if it's the first alarm for the event
            if not detected_events[next_event_index]:
                first_alarm_detected_distance[next_event_index] = max(
                    next_event_index - i, first_alarm_detected_distance[next_event_index]
                )

            # if we only consider the first alarm per event and its already caught, we skip
            if not (one_true_per_event and detected_events[next_event_index]):
                true_alarm_arr[i] = True
                true_alarm_count += 1

                # mark the event as detected
                detected_events[next_event_index] = True
                detected_distance[next_event_index] = max(
                    next_event_index - i, detected_distance[next_event_index]
                )

            else:
                # remove the raised alarm from the count
                # for the case where we only consider one per event
                # we leave the silencing and bool in place
                # last_alarm_index = prev_last_alarm_index # reset the last alarm to the first one
                raised_alarm_count -= 1

    detected_events_count = np.sum(detected_events)
    first_alarm_detected_distance_sum = np.sum(first_alarm_detected_distance)
    detection = (
        detected_events,
        detected_events_count,
        detected_distance,
        first_alarm_detected_distance_sum,
    )

    return raised_alarm_arr, raised_alarm_count, true_alarm_arr, true_alarm_count, detection


@numba.njit
def compute_alarms_at_thresholds_priority(
    inevent_arr: np.ndarray,
    pred_arr: np.ndarray,
    labeled_point_arr: np.ndarray,
    tau_threshs: np.ndarray,
    silence_time_steps: int = 30 // 5,
    reset_time_steps: int = 25 // 5,
    window_time_steps: int = 480 // 5,
    one_true_per_event: bool = False,
    merge_time_steps: int = 0,
    mask_reset_time_steps: bool = False,
    silencing_func: callable = base_silencing_function,
):
    """
    Compute alarms at different thresholds

    Parameter
    ---------
    one_true_per_event: bool
        whether to only consider one true alarm per event i.e. only the first alarm
        will be a positive true alarm, subsequent ones for the same event
        are neither false nor true
    mask_reset_time_steps: bool
        Whether to mask the time steps covered by the reset, if true, we first step through the time series
        and for all events we mask the time steps covered by the reset. This is done to avoid that we count
        events that are undetectable due to the reset as missed events. However, for the undetectable events
        the reset time is still applied as during deployment the event will still happen. The underlying assumption
        is, that the reset time merely covers a period of uncertainty, of what is actually a single event.
    """

    num_thresholds = tau_threshs.shape[0]
    results = np.empty((num_thresholds, 4), dtype=np.int64)

    # Mask time-points covered by reset
    if mask_reset_time_steps:
        labeled_point_arr = mask_reset_time_steps_to_unlabeled(
            inevent_arr, labeled_point_arr, reset_time_steps
        )

    # event merging
    if merge_time_steps > 0:
        endevent_arr_tmp, _ = compute_endevent_array(inevent_arr)
        inevent_arr = event_merge(inevent_arr, endevent_arr_tmp, merge_steps=merge_time_steps)

    onset_arr, next_event_indeces, event_counter, inevent_arr_onset_adj = compute_onset_array(
        inevent_arr,
        labeled_point_arr=labeled_point_arr,
        window_time_steps=window_time_steps,
        drop_events_only_full_window=False,
    )
    _, next_endevent_indeces = compute_endevent_array(inevent_arr_onset_adj)

    for i, tau in enumerate(tau_threshs):

        _, raised_alarm_count, _, true_alarm_count, detection = compute_alarms_priority(
            inevent_arr_onset_adj,
            onset_arr,
            next_event_indeces,
            next_endevent_indeces,
            pred_arr,
            labeled_point_arr,
            tau,
            reset_time_steps=reset_time_steps,
            silence_time_steps=silence_time_steps,
            window_time_steps=window_time_steps,
            one_true_per_event=one_true_per_event,
            detect_full_past_window=False,
            silencing_function=silencing_func,
        )

        _, detected_events_count, _, first_alarm_detected_distance = detection
        results[i, :] = (
            true_alarm_count,
            raised_alarm_count,
            detected_events_count,
            first_alarm_detected_distance,
        )

    return results, event_counter


# ----------------------------------------------------------------------------------------
#
# Score Computation over a dataset of patients
#
# ----------------------------------------------------------------------------------------
def compute_batch_patient_scores_priority(
    patient_dfs: list[pd.DataFrame],
    priority_func: callable,
    silencing_func: callable,
    tau_threshs: np.ndarray,
    survival_pred_horizons_steps: list[int],
    silence_time_steps: int,
    reset_time_steps: int,
    window_time_steps: int,
    one_true_per_event: bool = False,
    disable_tqdm: bool = False,
    merge_time_steps: int = 0,
    mask_reset_time_steps: bool = False,
) -> pd.DataFrame:
    """
    Compute the scores for a batch of patients:
    - True/False Alarms, Missed/Catched Events

    Parameter
    ---------
    mask_reset_time_steps: bool
        Whether to mask the time steps covered by the reset, if true, we first step through the time series
        and for all events we mask the time steps covered by the reset. This is done to avoid that we count
        events that are undetectable due to the reset as missed events. However, for the undetectable events
        the reset time is still applied as during deployment the event will still happen. The underlying assumption
        is, that the reset time merely covers a period of uncertainty, of what is actually a single event.
    """
    if mask_reset_time_steps:
        logging.info(
            f"[{EventEvaluation.__name__}:compute_batch_patient_scores_priority] Masking reset time steps is turned on"
        )

    patient_scores = []
    score_columns = [f"PredScore_{horizon_steps}" for horizon_steps in survival_pred_horizons_steps]

    for patient_df in tqdm(patient_dfs, disable=disable_tqdm):

        # compute labeled timepoint array
        time_point_labels = patient_df.TimeLabel.values
        labeled_point_arr = np.logical_not(
            np.logical_or(time_point_labels == -1, np.isnan(time_point_labels))
        )

        pred_scores = patient_df[score_columns].values
        pred_arr = priority_func(pred_scores)
        assert pred_arr.shape == pred_scores.shape

        scores = compute_alarms_at_thresholds_priority(
            patient_df.InEvent.values,
            pred_arr,
            labeled_point_arr,
            tau_threshs,
            silence_time_steps=silence_time_steps,
            reset_time_steps=reset_time_steps,
            window_time_steps=window_time_steps,
            one_true_per_event=one_true_per_event,
            merge_time_steps=merge_time_steps,
            mask_reset_time_steps=mask_reset_time_steps,
            silencing_func=silencing_func,
        )
        patient_scores.append(scores)

    return patient_scores