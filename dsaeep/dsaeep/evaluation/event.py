# ========================================
#
# Event based evaluation methods
#
# ========================================
import logging
import pickle
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from os import listdir, makedirs
from os.path import exists, join
from pathlib import Path
from typing import Any

import gin
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.stats as stats
import seaborn as sns
import torch
import wandb
from sklearn.metrics import auc
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsaeep.data.hirid.constants import (
    BATCH_PARQUET_PATTERN,
    DATETIME,
    PID,
    REL_DATETIME,
)
from dsaeep.data.utils import MarkedDataset
from dsaeep.pipeline import PipelineState, StatefulPipelineStage
from dsaeep.train.sequence import TabularWrapper
from dsaeep.train.utils import gin_config_to_readable_dictionary

# numba.config.THREADING_LAYER = 'omp'


# ================================================================================================
#
# Event Evaluation with NumPy and Numba
#
# ================================================================================================
@numba.njit()
def running_min(x):
    rmin = x[0]
    y = np.empty_like(x)
    for i, val in enumerate(x):
        if val < rmin:
            rmin = val
        y[i] = rmin
    return y


@numba.njit()
def event_merge(inevent_arr: np.ndarray, endevent_arr: np.ndarray, merge_steps: int):
    """
    Merge events that are closer than merge_steps

    Parameters
    ----------
    inevent_arr: np.ndarray
        Bool array
    endevent_arr: np.ndarray
        Bool array
    merge_steps: int
        If events are less than merge steps apart, they are merged
        by setting the flags as if they are one event


    """
    inevent_arr = inevent_arr.copy()  # we operate on and return a copy
    prior_step_inevent = False

    # as soon as we passed an event, i can be larger than last_event_end
    last_event_end = inevent_arr.shape[0] + 1

    for i, inevent in enumerate(inevent_arr):

        # this is an event start and we have passed
        # the end of a previous event
        if inevent and not prior_step_inevent and i > last_event_end:
            # check if we are within the merge steps
            gap = i - last_event_end
            if gap <= merge_steps:
                # set the end of the previous event to the end of this event
                inevent_arr[last_event_end:i] = True

        # we are at the end of an event
        if endevent_arr[i]:
            last_event_end = i

        prior_step_inevent = inevent

    return inevent_arr


@numba.njit
def compute_onset_array(
    inevent_arr: np.ndarray,
    labeled_point_arr: np.ndarray,
    window_time_steps: int = 480 // 5,  # 8 hours (480min) on a 5 minute grid
    drop_events_only_full_window: bool = False,
):
    """
    Mark the first timepoint of any event in the onset_arr.
    If the time series starts with an event, it is not considered
    as prior to it we have no chance of predicting it.

    Parameter
    ---------
    inevent_arr: np.ndarray
        Bool array if we are inside an event
    labeled_point_arr: np.ndarray
        Bool array if the timepoint is labeled
    window_time_steps: int
        Number of time steps to consider for the alarm window / the horizon
        If the window before is full of unlabelled points, we do not mark the onset
    drop_events_only_full_window: bool
        If True, we drop events only if the entire window before them is full
        of unlabeled points. If False, we only look until the end of the last event; thus
        if the last even is closer than window_time_steps, this causes diff. behavior.
        Only use for e.g. reproducing CircEWS results.
    """
    # we mark the start as being inevent
    # s.t. if the TS starts with an event
    # it's beginning is not marked
    # and we thus do not count it as a detectable event
    prior_step_inevent = True
    event_counter = 0

    # counts subsequent unlabeled points
    # if we have window_time_steps of unlabeled points
    # before a potential event, we could not detect it
    # thus we do not mark it's beginning
    unlabeled_point_counter = 0

    seq_len = inevent_arr.shape[0]
    out_of_bounds_index = seq_len + 1
    last_endevent_index = out_of_bounds_index

    onset_arr = np.zeros(inevent_arr.shape[0], dtype=numba.types.bool_)

    # storing adjusted inevent array values
    # if we have only unlabeled points before an event
    inevent_adj_arr = np.zeros(inevent_arr.shape[0], dtype=numba.types.bool_)
    # start with skipping the first event if there is one in the beginning
    skip_event = inevent_arr[0]

    for i, inevent in enumerate(inevent_arr):

        if not labeled_point_arr[i]:
            unlabeled_point_counter += 1

        # we are in event and prior step was not
        # we mark the onset
        # unless there are only unlabeled points before in the window
        entered_event = inevent and not prior_step_inevent
        if entered_event and (
            (
                unlabeled_point_counter < window_time_steps and unlabeled_point_counter < i
            )  # less unlabeled points than window or time past
            and (
                unlabeled_point_counter < i - last_endevent_index
                or last_endevent_index == out_of_bounds_index
                or drop_events_only_full_window
            )
        ):  # first event or some labeled points since last event
            onset_arr[i] = True
            event_counter += 1

        # this is a skipped event due to unlabeled points before it
        elif entered_event and (
            (
                unlabeled_point_counter >= window_time_steps or unlabeled_point_counter >= i
            )  # more/equal unlabeled points than window or time past
            or (
                unlabeled_point_counter >= i - last_endevent_index
                and last_endevent_index != out_of_bounds_index
                and not drop_events_only_full_window
            )
        ):  # or it's not the first event and there are no labeled points since last event
            skip_event = True

        if inevent and not skip_event:
            inevent_adj_arr[i] = True

        # we leave an event that was considered
        # NOTE: because of the way `prior_step_inevent` is init.
        # this will set `last_endevent_index` to 0 upon first iteration
        # if the first time point is not an event, which leads to correct execution
        # as distances are computed w.r.t. 0 i.e. the start of the time series
        if prior_step_inevent and not inevent and not skip_event:
            last_endevent_index = i

        prior_step_inevent = inevent

        # if we are at a labeled point and the previous one was not
        # we reset the labeled point counter
        # as events imply an unlabeled point, this will only reset
        # outside of events
        if i > 0 and labeled_point_arr[i] and not labeled_point_arr[i - 1]:
            unlabeled_point_counter = 0

        # we leave a skipped event, reset the skip event flag
        # and the unlabeled point counter
        if not inevent and skip_event:
            skip_event = False

    event_start_indeces = np.where(onset_arr, np.arange(inevent_arr.shape[0]), out_of_bounds_index)
    next_event_indeces = running_min(event_start_indeces[::-1])[::-1]

    return onset_arr, next_event_indeces, event_counter, inevent_adj_arr


@numba.njit
def compute_endevent_array(inevent_arr: np.ndarray):
    """
    Mark the last timepoint of an event
    """
    prior_step_inevent = False
    endevent_arr = np.zeros(inevent_arr.shape[0], dtype=numba.types.bool_)
    # endevent_arr = np.zeros(inevent_arr.shape[0], dtype=bool)

    seq_len = inevent_arr.shape[0]
    out_of_bounds_index = seq_len + 1

    for i in range(inevent_arr.shape[0] - 1, -1, -1):

        inevent = inevent_arr[i]
        if inevent and not prior_step_inevent:
            endevent_arr[i] = True

        prior_step_inevent = inevent

    event_end_indeces = np.where(endevent_arr, np.arange(inevent_arr.shape[0]), out_of_bounds_index)
    next_eventend_indeces = running_min(event_end_indeces[::-1])[::-1]

    return endevent_arr, next_eventend_indeces


@numba.njit
def mask_reset_time_steps_to_unlabeled(
    inevent_arr: np.ndarray,
    labeled_point_arr: np.ndarray,
    reset_time_steps: int,
) -> np.ndarray:
    """
    Masks all reset time steps after an event as unlabeled
    so we do not raise alarms and also do not consider
    events which become undetectable due to the reset.
    This operation is done in-place on the `labeled_point_arr`.

    Parameter
    ---------
    inevent_arr: np.ndarray
        Bool array if we are inside an event
    labeled_point_arr: np.ndarray
        Bool array if the timepoint is labeled
        Changes are done in-place
    reset_time_steps: int
        Number of time steps to reset the alarm after an event
        (e.g. for 20min reset, we mask 4 timepoints after the event)
    """
    seq_len = inevent_arr.shape[0]
    inevent = False

    for i in range(seq_len):
        # we enter an event
        if not inevent and inevent_arr[i]:
            inevent = True

        # we exit an event, we mask the reset time steps
        elif inevent and not inevent_arr[i]:
            inevent = False

            # mask the reset time steps
            # this includes the right boundary in the reset (e.g. for 20min reset, we mask 4 timepoints after the event)
            # i here is already the index of the first time step after the event
            labeled_point_arr[i : i + reset_time_steps] = False

    return labeled_point_arr


@numba.njit
def compute_alarms(
    inevent_arr: np.ndarray,
    onset_arr: np.ndarray,
    next_event_indeces: np.ndarray,
    next_endevent_indeces: np.ndarray,
    pred_arr: np.ndarray,
    labeled_point_arr: np.ndarray,
    tau_thresh: float,
    silence_time_steps: int = 30 // 5,
    reset_time_steps: int = 25 // 5,
    window_time_steps: int = 480 // 5,
    one_true_per_event: bool = False,
    detect_full_past_window: bool = False,
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
    pred_arr: np.ndarray
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

    pot_alarm_arr = pred_arr >= tau_thresh
    silenced_reset_loc_arr = np.zeros(seq_len, dtype=numba.types.bool_)

    raised_alarm_arr = np.zeros(seq_len, dtype=numba.types.bool_)
    true_alarm_arr = np.zeros(seq_len, dtype=numba.types.bool_)
    detected_events = np.zeros(seq_len, dtype=numba.types.bool_)
    detected_distance = np.zeros(seq_len, dtype=numba.types.int64)
    first_alarm_detected_distance = np.zeros(seq_len, dtype=numba.types.int64)
    raised_alarm_count = 0
    true_alarm_count = 0

    last_alarm_index = -1
    last_endevent_index = 0

    # the last timepoint does not need to be processed
    # as we do not raise an alarm if the stay is over
    for i in range(seq_len - 1):

        # we are at an event boundary, perform the reset
        # this includes the right boundary in the reset (e.g. for 20min reset, we mask 4 timepoints after the event)
        if i == next_endevent_indeces[i]:
            silenced_reset_loc_arr[i + 1 : i + reset_time_steps + 1] = True
            last_endevent_index = i

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

        # we step through because:
        # - this is anyway not an alarm location
        # - the location is silenced
        # - the location is not a labeled point
        if not pot_alarm_arr[i] or silenced_reset_loc_arr[i] or not labeled_point_arr[i]:
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

        # raise alarm and silence
        # prev_last_alarm_index = last_alarm_index
        last_alarm_index = i
        raised_alarm_arr[i] = True
        raised_alarm_count += 1

        # silence the alarm for silencing period
        # or until the next event
        # because as we enter an event the alarm is being reset
        # this includes the right boundary in the silencing (e.g. for 25min silencing, we mask 5 timepoints after the alarm)
        silence_end_index = min(i + silence_time_steps + 1, next_event_indeces[i] + 1)
        silenced_reset_loc_arr[i + 1 : silence_end_index] = True

        # check if the alarm is a True Alarm
        # - if the next event index is out of bounds, there will be no more event
        # - if the next event index is within the window, the alarm is a true alarm
        next_event_index = next_event_indeces[i]
        if next_event_index != out_of_bounds_index and i + window_time_steps >= next_event_index:

            if not detected_events[next_event_index]:
                first_alarm_detected_distance[next_event_index] = max(
                    next_event_index - i, first_alarm_detected_distance[next_event_index]
                )

            # if we only consider the first alarm per event and its already caught, we skip
            if not (one_true_per_event and detected_events[next_event_index]):
                true_alarm_arr[i] = True
                true_alarm_count += 1

                # mark the event as detected
                # next_event_index = next_event_indeces[i]
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
def compute_alarms_at_thresholds(
    inevent_arr: np.ndarray,
    pred_arr: np.ndarray,
    labeled_point_arr: np.ndarray,
    tau_threshs: np.ndarray,
    silence_time_steps: int = 30 // 5,
    reset_time_steps: int = 25 // 5,
    window_time_steps: int = 480 // 5,
    one_true_per_event: bool = False,
    merge_time_steps: int = 0,
    detect_full_past_window: bool = False,
    drop_events_only_full_window: bool = False,
    mask_reset_time_steps: bool = False,
):
    """
    Compute alarms at different thresholds

    Parameter
    ---------
    one_true_per_event: bool
        whether to only consider one true alarm per event i.e. only the first alarm
        will be a positive true alarm, subsequent ones for the same event
        are neither false nor true
    detect_full_past_window: bool
        In the CircEWS paper (caption Figure 2) and also here: https://github.com/ratschlab/circEWS/blob/b2b1f00dac4f5d46856a2c7abe2ca4f12d4c612d/evaluation/precision_recall.py#LL328C6-L328C6
        it is stated that an event is considered caught if in the full 8h window before it there has been an alarm.
        Per default we do not consider this and only consider either the full 8h window or (if shorter) only the window
        to the last event end.
    drop_events_only_full_window: bool
        If True, we drop events only if the entire window before them is full
        of unlabeled points. If False, we only look until the end of the last event; thus
        if the last even is closer than window_time_steps, this causes diff. behavior.
        Only use for e.g. reproducing CircEWS results.
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
        drop_events_only_full_window=drop_events_only_full_window,
    )
    _, next_endevent_indeces = compute_endevent_array(inevent_arr_onset_adj)

    for i, tau in enumerate(tau_threshs):
        _, raised_alarm_count, _, true_alarm_count, detection = compute_alarms(
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
            detect_full_past_window=detect_full_past_window,
        )

        _, detected_events_count, _, first_alarm_detected_distance = detection
        results[i, :] = (
            true_alarm_count,
            raised_alarm_count,
            detected_events_count,
            first_alarm_detected_distance,
        )

    return results, event_counter


@numba.njit
def compute_alarm_distances(
    inevent_arr: np.ndarray,
    pred_arr: np.ndarray,
    labeled_point_arr: np.ndarray,
    tau: float,
    silence_time_steps: int = 30 // 5,
    reset_time_steps: int = 25 // 5,
    window_time_steps: int = 480 // 5,
    merge_time_steps: int = 0,
    detect_full_past_window: bool = False,
    drop_events_only_full_window: bool = False,
    mask_reset_time_steps: bool = False,
) -> tuple[np.ndarray, tuple, int, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Compute alarms give threshold and collect distance data

    Parameter
    ---------
    detect_full_past_window: bool
        In the CircEWS paper (caption Figure 2) and also here: https://github.com/ratschlab/circEWS/blob/b2b1f00dac4f5d46856a2c7abe2ca4f12d4c612d/evaluation/precision_recall.py#LL328C6-L328C6
        it is stated that an event is considered caught if in the full 8h window before it there has been an alarm.
        Per default we do not consider this and only consider either the full 8h window or (if shorter) only the window
        to the last event end.
    drop_events_only_full_window: bool
        If True, we drop events only if the entire window before them is full
        of unlabeled points. If False, we only look until the end of the last event; thus
        if the last even is closer than window_time_steps, this causes diff. behavior.
        Only use for e.g. reproducing CircEWS results.
    mask_reset_time_steps: bool
        Whether to mask the time steps covered by the reset, if true, we first step through the time series
        and for all events we mask the time steps covered by the reset. This is done to avoid that we count
        events that are undetectable due to the reset as missed events. However, for the undetectable events
        the reset time is still applied as during deployment the event will still happen. The underlying assumption
        is, that the reset time merely covers a period of uncertainty, of what is actually a single event.
    """

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
        drop_events_only_full_window=drop_events_only_full_window,
    )
    endevent_arr, next_endevent_indeces = compute_endevent_array(inevent_arr_onset_adj)

    raised_alarm_arr, _, _, _, detection = compute_alarms(
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
        one_true_per_event=False,
        detect_full_past_window=detect_full_past_window,
    )

    onset_data = (onset_arr, next_event_indeces)
    endevent_data = (endevent_arr, next_endevent_indeces)

    return raised_alarm_arr, detection, event_counter, onset_data, endevent_data


@gin.configurable("SilencingConfig")
@dataclass
class SilencingConfig:
    """
    Configuration for the silencing of alarms

    window_size_min: int (minutes)
            Size of the prediction window in seconds (i.e. the event horizon)
    silence_time_min: int (minutes)
        Time in minutes for the silencing period
    reset_time_min: int (minutes)
        Time in minutes to reset the alarm
    """

    window_size_min: int = 480
    reset_time_min: int = 25

    silence_time_min: int = 30
    silence_optimize: bool = False
    silence_optimize_mins: list[int] = field(
        default_factory=lambda: [10, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    )


@gin.configurable("EventEvaluation", denylist=["state"])
class EventEvaluation(StatefulPipelineStage):
    """
    Event Evaluation Pipeline Stage
    """

    name = "Event Evaluation"

    def __init__(
        self,
        state: PipelineState,
        patient_endpoint_data_dir: Path,
        num_workers: int = 1,
        silence_config: SilencingConfig = SilencingConfig(),
        num_thresholds=100,
        event_column: str = "circ_failure_status",
        accelerator: str = "cpu",
        batch_size: int = 4,
        use_pred_cache: bool = False,
        use_endpoint_cache: bool = False,
        subsample_parts: int = -1,
        store_scores: bool = False,
        target_recalls: list[float] = [0.8, 0.9, 0.95],
        lateness_target_recalls: list[float] = [0.25, 0.5, 0.75, 0.9],
        lateness_target_silencings: list[int] = [30, 120, 360],
        lateness_silencing_recall: float = 0.8,
        reset_times_performance: list[int] = [5, 15, 25, 45, 60, 120],
        merge_time_min: int = 0,
        detect_full_past_window: bool = False,
        drop_events_only_full_window: bool = False,
        mask_reset_time_steps: bool = False,
        delta_t: int = 0,
        feature_columns: list[str] = None,
        step_size_min: int = 5,
        debug: bool = False,
        **kwargs,
    ):
        """
        Constructor for `EventPrecisionRecallReference`

        Parameter
        ---------
        state: PipelineState
            The pipeline state
        patient_endpoint_data_dir: Path
            Path to the patient endpoint data
        num_workers: int
            Number of workers to use for parallel processing
        silence_config: SilencingConfig
            Configuration for the silencing of alarms
        num_thresholds: int
            Number of thresholds to compute the scores for
        subsample_parts: int
            Number of endpoint parts to subsample from the dataset
        store_scores: bool
            Whether to store the scores in the results folder
        max_gap_to_merge: int (minutes)
            Maximum gap in minutes betw. events to merge
        delta_t: int (minutes)
            Minimum time past before we consider events and
            gap before events were we do not consider predictions
        feature_columns: list[int]
            feature columns to use for tabular wrappers / sklearn models
        debug: bool
            Whether to run in debug mode, e.g. stores additional data
        merge_time_min: int
            Time in minutes to merge events that are closer than this
        detect_full_past_window: bool
            In the CircEWS paper (caption Figure 2) and also here: https://github.com/ratschlab/circEWS/blob/b2b1f00dac4f5d46856a2c7abe2ca4f12d4c612d/evaluation/precision_recall.py#LL328C6-L328C6
            it is stated that an event is considered caught if in the full 8h window before it there has been an alarm.
            Per default we do not consider this and only consider either the full 8h window or (if shorter) only the window
            to the last event end.
        drop_events_only_full_window: bool
            If True, we drop events only if the entire window before them is full
            of unlabeled points. If False, we only look until the end of the last event; thus
            if the last even is closer than window_time_steps, this causes diff. behavior.
            Only use for e.g. reproducing CircEWS results.
        mask_reset_time_steps: bool
            Whether to mask the time steps covered by the reset, if true, we first step through the time series
            and for all events we mask the time steps covered by the reset. This is done to avoid that we count
            events that are undetectable due to the reset as missed events. However, for the undetectable events
            the reset time is still applied as during deployment the event will still happen. The underlying assumption
            is, that the reset time merely covers a period of uncertainty, between what is actually a single event.
        """
        super().__init__(state, num_workers=num_workers, **kwargs)

        self.delta_t = delta_t
        self.merge_time_min = merge_time_min
        assert self.delta_t == 0, "Currently only delta_t=0 is supported"

        self.silence_config = silence_config
        self.window_size_min = silence_config.window_size_min
        self.silence_time_min = silence_config.silence_time_min
        self.reset_time_min = silence_config.reset_time_min
        self.step_size_min = step_size_min
        logging.info(
            f"[{self.__class__.__name__}] Window size: {self.window_size_min} minutes (i.e. {self.window_size_min / 60} hours)"
        )
        logging.info(
            f"[{self.__class__.__name__}] Silencing time: {self.silence_time_min} minutes (i.e. {self.silence_time_min // self.step_size_min} steps)"
        )
        logging.info(
            f"[{self.__class__.__name__}] Reset time: {self.reset_time_min} minutes (i.e. {self.reset_time_min // self.step_size_min} steps)"
        )
        if self.merge_time_min > 0:
            logging.warning(
                f"[{self.__class__.__name__}] Merging events within {self.merge_time_min} minutes"
            )

        self.silence_time_sec = self.silence_time_min * 60
        self.max_sec = (self.delta_t + self.window_size_min) * 60
        self.min_sec = self.delta_t * 60

        self.num_thresholds = num_thresholds
        min_pred_score = 0.0
        max_pred_score = 1.0
        # thresholds = np.log(np.linspace(1, 1000, num=num_thresholds)) / np.log(1000)
        thresholds = np.linspace(0, 1, num=num_thresholds)

        self.thresholds = min_pred_score + thresholds * (max_pred_score - min_pred_score)

        self.patient_endpoint_data_dir = Path(patient_endpoint_data_dir)
        self.event_column = event_column
        self.subsample_parts = subsample_parts
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.use_pred_cache = use_pred_cache
        self.use_endpoint_cache = use_endpoint_cache
        self.store_scores = store_scores
        self.feature_columns = feature_columns
        self.debug = debug

        if self.debug:
            logging.info(f"[{self.__class__.__name__}] Running in debug mode")

        self.target_recalls = target_recalls
        self.lateness_target_recalls = lateness_target_recalls
        self.lateness_target_silencings = lateness_target_silencings
        self.lateness_silencing_recall = lateness_silencing_recall
        self.reset_times_performance = reset_times_performance
        self.detect_full_past_window = detect_full_past_window
        if self.detect_full_past_window:
            logging.warning(
                f"[{EventEvaluation.__name__}:init] Detecting full past window is turned on: use only for reproducing CircEWS"
            )

        self.drop_events_only_full_window = drop_events_only_full_window
        if self.drop_events_only_full_window:
            logging.warning(
                f"[{EventEvaluation.__name__}:init] Dropping events only full window is turned on: use only for reproducing CircEWS"
            )

        self.mask_reset_time_steps = mask_reset_time_steps
        if self.mask_reset_time_steps:
            logging.info(f"[{EventEvaluation.__name__}:init] Masking reset time steps is turned on")

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        return False

    @staticmethod
    def compute_batch_patient_scores(
        patient_dfs: list[pd.DataFrame],
        tau_threshs: np.ndarray,
        silence_time_steps: int,
        reset_time_steps: int,
        window_time_steps: int,
        one_true_per_event: bool = False,
        disable_tqdm: bool = False,
        merge_time_steps: int = 0,
        detect_full_past_window: bool = False,
        drop_events_only_full_window: bool = False,
        mask_reset_time_steps: bool = False,
    ) -> pd.DataFrame:
        """
        Compute the scores for a batch of patients:
        - True/False Alarms, Missed/Catched Events

        Parameter
        ---------
        detect_full_past_window: bool
            In the CircEWS paper (caption Figure 2) and also here: https://github.com/ratschlab/circEWS/blob/b2b1f00dac4f5d46856a2c7abe2ca4f12d4c612d/evaluation/precision_recall.py#LL328C6-L328C6
            it is stated that an event is considered caught if in the full 8h window before it there has been an alarm.
            Per default we do not consider this and only consider either the full 8h window or (if shorter) only the window
            to the last event end.
        drop_events_only_full_window: bool
            If True, we drop events only if the entire window before them is full
            of unlabeled points. If False, we only look until the end of the last event; thus
            if the last even is closer than window_time_steps, this causes diff. behavior.
            Only use for e.g. reproducing CircEWS results.
        mask_reset_time_steps: bool
            Whether to mask the time steps covered by the reset, if true, we first step through the time series
            and for all events we mask the time steps covered by the reset. This is done to avoid that we count
            events that are undetectable due to the reset as missed events. However, for the undetectable events
            the reset time is still applied as during deployment the event will still happen. The underlying assumption
            is, that the reset time merely covers a period of uncertainty, of what is actually a single event.
        """
        if detect_full_past_window:
            logging.warning(
                f"[{EventEvaluation.__name__}:compute_batch_patient_scores] Detecting full past window is turned on: use only for reproducing CircEWS"
            )

        if drop_events_only_full_window:
            logging.warning(
                f"[{EventEvaluation.__name__}:compute_batch_patient_scores] Dropping events only full window is turned on: use only for reproducing CircEWS"
            )

        if mask_reset_time_steps:
            logging.info(
                f"[{EventEvaluation.__name__}:compute_batch_patient_scores] Masking reset time steps is turned on"
            )

        patient_scores = []

        for patient_df in tqdm(patient_dfs, disable=disable_tqdm):
            # compute labeled timepoint array
            time_point_labels = patient_df.TimeLabel.values
            labeled_point_arr = np.logical_not(
                np.logical_or(time_point_labels == -1, np.isnan(time_point_labels))
            )

            scores = compute_alarms_at_thresholds(
                patient_df.InEvent.values,
                patient_df.PredScore.values,
                labeled_point_arr,
                tau_threshs,
                silence_time_steps=silence_time_steps,
                reset_time_steps=reset_time_steps,
                window_time_steps=window_time_steps,
                one_true_per_event=one_true_per_event,
                merge_time_steps=merge_time_steps,
                detect_full_past_window=detect_full_past_window,
                drop_events_only_full_window=drop_events_only_full_window,
                mask_reset_time_steps=mask_reset_time_steps,
            )
            patient_scores.append(scores)

        return patient_scores

    @staticmethod
    def optimize_silencing(
        patient_dfs: list[pd.DataFrame],
        tau_threshs: np.ndarray,
        window_time_steps: int,
        reset_time_steps: int,
        silence_times_mins: list[int],
        merge_time_min: int = 0,
        one_true_per_event: bool = False,
        mask_reset_time_steps: bool = False,
        step_size_min: int = 5,
    ) -> tuple[tuple[int, float], plt.Axes]:
        """
        Returns a brute-forced best silencing time
        optimizing Event-AuPRC.
        """
        silencing_scores = []

        logging.info(
            f"[{EventEvaluation.__name__}] Optimizing silencing time: {len(silence_times_mins)} silencing times"
        )
        for silene_time_min in tqdm(silence_times_mins):
            # Compute scores over dataset
            patient_scores = EventEvaluation.compute_batch_patient_scores(
                patient_dfs=patient_dfs,
                tau_threshs=tau_threshs,
                silence_time_steps=silene_time_min // step_size_min,
                reset_time_steps=reset_time_steps,
                window_time_steps=window_time_steps,
                one_true_per_event=one_true_per_event,
                merge_time_steps=merge_time_min // step_size_min,
                disable_tqdm=True,
                mask_reset_time_steps=mask_reset_time_steps,
            )

            precisions, recalls, _ = EventEvaluation.compute_precision_recall(
                patient_scores, tau_threshs
            )

            pr_auc, _, _ = EventEvaluation.plot_precision_recall(precisions, recalls, auc_only=True)

            silencing_scores.append((silene_time_min, pr_auc))

        best_silencing = max(silencing_scores, key=lambda x: x[1])
        logging.info(
            f"[{EventEvaluation.__name__}] Best silencing time: {best_silencing[0]} min, AuPRC: {best_silencing[1]:.4f}"
        )

        plot_x = [x[0] for x in silencing_scores]
        plot_y = [x[1] for x in silencing_scores]
        plt.clf()
        pr_plot = sns.lineplot(x=plot_x, y=plot_y)

        pr_plot.set_title(f"Silencing Time Optimization")
        pr_plot.set(xlabel="Silencing Time (min)", ylabel="Event AuPRC")

        pr_plot.set_xlim(0, max(plot_x))
        pr_plot.set_ylim(0.0, 1.0)

        return best_silencing, pr_plot

    @staticmethod
    def find_tau_for_recall(
        precisions: np.ndarray, recalls: np.ndarray, thresholds: np.ndarray, target_recall: float
    ) -> tuple[float, float, float]:

        sort_index = np.argsort(recalls)
        recalls = recalls[sort_index]
        precisions = precisions[sort_index]
        thresholds = thresholds[sort_index]

        # find the first threshold where the recall is above the target recall
        # and the precision is not 0
        tau = thresholds[np.logical_and(recalls >= target_recall, precisions > 0)][0]
        tau_index = np.where(thresholds == tau)[0][0]

        recall = recalls[tau_index]
        precision = precisions[tau_index]

        return tau, recall, precision

    @staticmethod
    def compute_recall_event_distance(
        patient_dfs: list[pd.DataFrame],
        tau: float,
        silence_time_steps: int,
        reset_time_steps: int,
        window_time_steps: int,
        hour_grid: np.ndarray,
        merge_time_steps: int,
        detect_full_past_window: bool = False,
        drop_events_only_full_window: bool = False,
        mask_reset_time_steps: bool = False,
    ):
        """
        Compute recall at different distances from the event

        Parameter
        ---------
        detect_full_past_window: bool
            In the CircEWS paper (caption Figure 2) and also here: https://github.com/ratschlab/circEWS/blob/b2b1f00dac4f5d46856a2c7abe2ca4f12d4c612d/evaluation/precision_recall.py#LL328C6-L328C6
            it is stated that an event is considered caught if in the full 8h window before it there has been an alarm.
            Per default we do not consider this and only consider either the full 8h window or (if shorter) only the window
            to the last event end.
        drop_events_only_full_window: bool
            If True, we drop events only if the entire window before them is full
            of unlabeled points. If False, we only look until the end of the last event; thus
            if the last even is closer than window_time_steps, this causes diff. behavior.
            Only use for e.g. reproducing CircEWS results.
        mask_reset_time_steps: bool
            Whether to mask the time steps covered by the reset, if true, we first step through the time series
            and for all events we mask the time steps covered by the reset. This is done to avoid that we count
            events that are undetectable due to the reset as missed events. However, for the undetectable events
            the reset time is still applied as during deployment the event will still happen. The underlying assumption
            is, that the reset time merely covers a period of uncertainty, of what is actually a single event.
        """
        recall_at_distance = np.zeros(len(hour_grid))
        total_events_at_distance = np.zeros(len(hour_grid))
        total_event_counter = 0
        total_detected_event = 0
        distance_to_event = []
        lateness_for_event = []
        event_gaps = []
        alarm_per_patient = []

        if detect_full_past_window or drop_events_only_full_window:
            logging.warning(
                f"[{EventEvaluation.__name__}:compute_recall_event_distance] Detecting full past window: {detect_full_past_window}"
            )
            logging.warning(
                f"[{EventEvaluation.__name__}:compute_recall_event_distance] Dropping events only full window: {drop_events_only_full_window}"
            )

        if mask_reset_time_steps:
            logging.info(
                f"[{EventEvaluation.__name__}:compute_recall_event_distance] Masking reset time steps is turned on"
            )

        for patient_df in patient_dfs:

            # compute labeled timepoint array
            time_point_labels = patient_df.TimeLabel.values
            labeled_point_arr = np.logical_not(
                np.logical_or(time_point_labels == -1, np.isnan(time_point_labels))
            )

            (
                raised_alarm_arr,
                detection,
                event_counter,
                onset_data,
                endevent_data,
            ) = compute_alarm_distances(
                patient_df.InEvent.values,
                patient_df.PredScore.values,
                labeled_point_arr,
                tau,
                silence_time_steps=silence_time_steps,
                reset_time_steps=reset_time_steps,
                window_time_steps=window_time_steps,
                merge_time_steps=merge_time_steps,
                detect_full_past_window=detect_full_past_window,
                drop_events_only_full_window=drop_events_only_full_window,
                mask_reset_time_steps=mask_reset_time_steps,
            )
            (
                detected_events,
                detected_events_count,
                detected_distance,
                first_alarm_detected_distance_sum,
            ) = detection

            # count all events
            total_event_counter += event_counter
            total_detected_event += detected_events_count
            alarm_per_patient.append(np.sum(raised_alarm_arr))

            onset_arr, next_event_start = onset_data
            endevent_arr, next_event_end = endevent_data

            last_event_end = 0
            onset_indeces = np.where(onset_arr)[0]
            endevent_indeces = np.where(endevent_arr)[0]
            raised_alarm_indeces = np.where(
                raised_alarm_arr, np.arange(raised_alarm_arr.shape[0]), -1
            )
            # if TS starts with an event
            # we do not consider it as we have no chance of predicting it
            # thus: drop first event
            if onset_arr[0]:
                logging.error(
                    f"[{EventEvaluation.__name__}:compute_recall_event_distance] TS starts with an event"
                )
                last_event_end = endevent_indeces[0]
                endevent_indeces = endevent_indeces[1:]
                onset_indeces = onset_indeces[1:]

            assert (
                len(endevent_indeces) == event_counter
            ), f"Event counter: {event_counter}, but end indeces: {endevent_indeces}"
            assert (
                len(onset_indeces) == event_counter
            ), f"Event counter: {event_counter}, but start indeces: {onset_indeces}"

            check_detected_events = 0
            lateness_values = []

            for onset_index, end_index in zip(onset_indeces, endevent_indeces):

                # Count the event as is and compute
                # its earliest possible detection
                window_or_start_distance_steps = min(onset_index, window_time_steps)
                window_size_pre_event_steps = (
                    (onset_index - last_event_end)
                    if not detect_full_past_window
                    else window_or_start_distance_steps
                )

                window_size_pre_event_hour = window_size_pre_event_steps / 12
                event_gaps.append(window_size_pre_event_hour)

                total_events_at_distance[np.argmax(window_size_pre_event_hour >= hour_grid)] += 1

                # Get all alarms in the window before
                lower_bound = (
                    max(last_event_end, onset_index - window_time_steps)
                    if not detect_full_past_window
                    else (onset_index - window_time_steps)
                )
                lower_bound = max(lower_bound, 0)

                window_raised_alarm_bool = raised_alarm_arr[lower_bound:onset_index]
                window_raised_alarm_count = window_raised_alarm_bool.sum()

                # set end_index of last event
                last_event_end = end_index

                # this is a missed event
                if window_raised_alarm_count == 0:
                    continue

                # this is a caught event
                check_detected_events += 1

                # get distance to furthest alarm
                window_raised_alarm_indeces = raised_alarm_indeces[lower_bound:onset_index]
                window_raised_alarm_indeces = window_raised_alarm_indeces[
                    window_raised_alarm_indeces != -1
                ]
                earliest_alarm_index = window_raised_alarm_indeces[0]

                dist = onset_index - earliest_alarm_index
                dist_hour = dist / 12  # convert to hours from steps

                # compute lateness (from window begin or end of last event)
                # and collect all the lateness values for later distribution plotting
                lateness = earliest_alarm_index - lower_bound
                lateness_hour = lateness / 12
                lateness_values.append(lateness_hour)

                # get first index where dist is larger than hour_grid
                hour_index = np.argmax(dist_hour >= hour_grid)
                recall_at_distance[hour_index] += 1

                distance_to_event.append(dist_hour)

            lateness_for_event.append(lateness_values)
            assert check_detected_events == detected_events_count

        recall_at_distance = np.cumsum(recall_at_distance)
        total_events_at_distance = np.cumsum(total_events_at_distance)
        assert recall_at_distance[-1] == total_detected_event
        assert total_events_at_distance[-1] == total_event_counter

        adjusted_recall_at_distance = recall_at_distance / total_events_at_distance
        recall_at_distance = recall_at_distance / total_event_counter

        distance_stats = (
            np.median(distance_to_event),
            np.mean(distance_to_event),
            np.std(distance_to_event),
        )

        return (
            recall_at_distance,
            adjusted_recall_at_distance,
            total_event_counter,
            total_detected_event,
            distance_stats,
            lateness_for_event,
            event_gaps,
        )

    @staticmethod
    def plot_recall_at_distance(
        recall_at_distance: list[np.ndarray],
        hour_grid: list[np.ndarray],
        target_recalls: list[float],
        precision_at_recall: list[float],
        silence_time_min: int,
        dist_stat_data: list[tuple[float, float]],
    ) -> plt.Axes:
        """
        Plot the recall at distance to an event
        """
        recall_at_distance_cat = []
        hour_grid_cat = []
        target_recalls_cat = []
        mean_dist = []

        zipped_iterator = zip(
            recall_at_distance, hour_grid, target_recalls, precision_at_recall, dist_stat_data
        )
        for rec, grid, target_recall, prec, stats in zipped_iterator:
            recall_at_distance_cat.append(rec)
            hour_grid_cat.append(grid)
            target_recalls_cat.extend(
                [
                    f"Rec={target_recall:.2f}, prec={prec:.2f}, mean dist={stats[1]:.2f}h"
                    for _ in range(len(grid))
                ]
            )
            mean_dist.extend([stats[1] for _ in range(len(grid))])

        recall_at_distance_cat = np.concatenate(recall_at_distance_cat)
        hour_grid_cat = np.concatenate(hour_grid_cat)
        data_df = pd.DataFrame(
            {
                "recall": recall_at_distance_cat,
                "hour": hour_grid_cat,
                "Recall / Precision": target_recalls_cat,
                "Mean Dist.": mean_dist,
            }
        )

        unique = data_df["Recall / Precision"].unique()
        palette = dict(zip(unique, sns.color_palette(palette="tab10", n_colors=len(unique))))

        fig, ax = plt.subplots(figsize=(10, 10))
        recall_plot = sns.lineplot(
            data=data_df, x="hour", y="recall", hue="Recall / Precision", palette=palette, ax=ax
        )
        sns.move_legend(recall_plot, "upper left", bbox_to_anchor=(1, 1))

        for key, value in palette.items():
            mean_dist_x = data_df[data_df["Recall / Precision"] == key]["Mean Dist."].iloc[0]
            recall_plot.axvline(x=mean_dist_x, color=value, linestyle="--", alpha=0.5)

        title = f"Recall approaching Events ({silence_time_min} min silencing)"
        recall_plot.set_title(title)
        recall_plot.set(xlabel="First Alarm Time before Event (h)", ylabel="Event Recall")

        recall_plot.set_xlim(hour_grid[0][0], hour_grid[0][-1])
        recall_plot.set_ylim(0.0, 1.0)

        return recall_plot

    @staticmethod
    def plot_lateness_distribution(
        lateness_data: list[np.ndarray],
        hour_grid_data: list[np.ndarray],
        target_recalls: list[float],
        precision_at_recall: list[float],
        tau_at_recall: list[float],
        silence_times_min: list[int],
        subtitle: str = "",
    ) -> plt.Axes:
        """
        Plot the distribution of lateness
        for fixed threshold and different silencing times
        """
        lateness_data_cat = []
        target_recalls_cat = []

        zipped_iterator = zip(
            lateness_data, target_recalls, precision_at_recall, tau_at_recall, silence_times_min
        )
        for lateness, target_recall, prec, tau, sil in zipped_iterator:
            lateness_data_cat.append(lateness)
            target_recalls_cat.extend(
                [
                    f"Rec={target_recall:.2f}, prec={prec:.2f}, tau={tau:.2f}, sil={sil}"
                    for _ in range(len(lateness))
                ]
            )
            # tau_at_recall_cat.extend([tau for _ in range(len(lateness))])

        lateness_data_cat = np.concatenate(lateness_data_cat)
        data_df = pd.DataFrame(
            {"lateness": lateness_data_cat, "Recall / Precision": target_recalls_cat}
        )

        unique = data_df["Recall / Precision"].unique()
        palette = dict(zip(unique, sns.color_palette(palette="tab10", n_colors=len(unique))))

        fig, ax = plt.subplots(figsize=(10, 10))
        lateness_plot = sns.displot(
            data=data_df,
            x="lateness",
            hue="Recall / Precision",
            palette=palette,
            ax=ax,
            kind="kde",
            fill=True,
            bw_adjust=0.5,
            common_norm=False,
        )
        # sns.move_legend(lateness_plot, "upper left", bbox_to_anchor=(1, 1))

        title = f"Lateness Distribution {subtitle}"
        lateness_plot.fig.suptitle(title)
        lateness_plot.set(xlabel="Lateness (hours)")

        lateness_plot.set(xlim=(hour_grid_data[0][-1], hour_grid_data[0][0]))
        lateness_plot.set(ylim=(0.0, 1.0))

        return lateness_plot

    @staticmethod
    def plot_time_gaps(time_gaps: list[float], cut_off: float = 16.0):
        """
        Plot the distribution of time gaps between events
        """

        time_gaps = list(filter(lambda x: x < cut_off, time_gaps))
        logging.info(f"[EventEvaluation] Smallest time gap: {int(np.min(time_gaps) * 60)} minutes")

        fig, ax = plt.subplots(figsize=(10, 10))
        time_gap_plot = sns.displot(data=time_gaps, ax=ax, kind="hist", bins=100)

        title = f"Time Gaps before Events (clipped at {cut_off} hours)"
        time_gap_plot.fig.suptitle(title)
        time_gap_plot.set(xlabel="Time Gap (hours)")

        plt.gcf().subplots_adjust(bottom=0.15)

        return time_gap_plot

    def load_patient_endpoint_data(self) -> list[pd.DataFrame]:
        """
        Load the patient endpoint data
        """
        # find .parquet files in the directory with rglob
        patient_parquet_file = next(self.patient_endpoint_data_dir.rglob("*.parquet"))
        patient_parquet_directory = patient_parquet_file.parent

        self.patient_endpoint_dataset = MarkedDataset(
            patient_parquet_directory, part_re=BATCH_PARQUET_PATTERN, force=False
        )

        patient_dfs = None

        endpoint_cache_file = Path(self.state.log_dir) / "predictions" / f"endpoint_cache.pkl"
        if self.use_endpoint_cache and endpoint_cache_file.exists():
            logging.warning(
                f"[{self.__class__.__name__}] Loading endpoint data from {endpoint_cache_file}"
            )
            with open(endpoint_cache_file, "rb") as f:
                patient_dfs = pickle.load(f)

        found_nans = False
        if patient_dfs is None:

            parts_list = self.patient_endpoint_dataset.list_parts()
            if self.subsample_parts > 0:
                parts_list = parts_list[: self.subsample_parts]

            logging.info(
                f"[{self.__class__.__name__}] Loading patient endpoint data: {patient_parquet_directory}"
            )
            logging.info(
                f"[{self.__class__.__name__}] Loading patient endpoint data from {len(parts_list)} parts"
            )

            patient_dfs = []
            keep_columns = [self.event_column, REL_DATETIME, PID]

            for part in tqdm(parts_list):
                patient_df = pd.read_parquet(part, columns=keep_columns)

                patient_df["InEvent"] = patient_df[self.event_column] > 0
                patient_df = patient_df.drop(columns=[self.event_column])

                # if there are nan's we fill them with False
                # if before it was False, an unkown state contains no event
                # if before it was True, an unkown state ends an event
                contains_nans = patient_df["InEvent"].isna().sum() > 0
                found_nans = found_nans or contains_nans
                if contains_nans:
                    patient_df["InEvent"].fillna(False, inplace=True)

                patient_ids = patient_df[PID].unique()
                grouped_patient_df = patient_df.groupby(PID)

                patient_df_chunk = [
                    grouped_patient_df.get_group(pid).set_index(REL_DATETIME) for pid in patient_ids
                ]

                patient_dfs.extend(patient_df_chunk)

        if found_nans:
            logging.warning(f"[{self.__class__.__name__}] Found NaNs in endpoint data")

        if self.use_endpoint_cache and not endpoint_cache_file.exists():
            logging.info(
                f"[{self.__class__.__name__}] Caching endpoint data to {endpoint_cache_file}"
            )
            endpoint_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(endpoint_cache_file, "wb") as f:
                pickle.dump(patient_dfs, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(
            f"[{self.__class__.__name__}] Loaded {len(patient_dfs)} patients endpoint dataframes"
        )

        return patient_dfs

    def load_patient_prediction_data(
        self, patient_dfs: list[pd.DataFrame], split: str = "val"
    ) -> list[pd.DataFrame]:
        """
        Load the patient prediction data

        Parameters
        ----------
        patient_dfs : list[pd.DataFrame]
            The patient dataframes
        """
        predictions_file = (
            Path(self.state.log_dir) / "predictions" / f"predictions_split_{split}.pkl"
        )
        if self.use_pred_cache and predictions_file.exists():
            logging.warning(
                f"[{self.__class__.__name__}] Loading predictions from {predictions_file}"
            )
            with open(predictions_file, "rb") as f:
                predictions = pickle.load(f)

        elif isinstance(self.state.model_wrapper, pl.LightningModule):
            assert self.state.dataset_class is not None, "No dataset_class set"
            dataset = self.state.dataset_class(
                self.state.data_path, split=split, return_ids=True, keep_only_valid_patients=False
            )
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

            trainer = pl.Trainer(
                accelerator=self.accelerator,
                devices=1 if self.accelerator == "gpu" else None,
                logger=False,
            )
            predictions = trainer.predict(self.state.model_wrapper, dataloader)

        elif isinstance(
            self.state.model_wrapper, TabularWrapper
        ):  # using `TabularWrapper` / Sklearn style model
            assert self.state.dataset_class is not None, "No dataset_class set"
            dataset = self.state.dataset_class(self.state.data_path, split=split, return_ids=True)

            rep, label, patient_ids = dataset.get_data_and_labels(
                columns=self.feature_columns, drop_unlabeled=False
            )
            preds = self.state.model_wrapper.predict(rep)[:, 1]

            # Split into patient > time-series
            preds_list = []
            label_list = []
            patient_ids_list = []
            logging.info(
                f"[{self.__class__.__name__}] Splitting predictions into patients with time-series"
            )
            for pid in tqdm(np.unique(patient_ids)):
                pid_mask = patient_ids == pid
                preds_list.append(preds[pid_mask])
                label_list.append(label[pid_mask])
                patient_ids_list.append(pid)

            # list for "1 batch"
            predictions = [(preds_list, label_list, patient_ids_list)]

        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported model wrapper: {type(self.state_model_wrapper)}"
            )

        # Cache predictions
        if self.use_pred_cache and not predictions_file.exists():
            predictions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(predictions_file, "wb") as f:
                pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)

        # load prediction batches and prepare for analysis
        if isinstance(self.state.model_wrapper, pl.LightningModule):
            preds = []
            time_labels = []
            patient_ids = []
            for batch in predictions:
                preds.append(batch[0])
                time_labels.append(batch[1])
                patient_ids.append(batch[2])

            preds_tensor = torch.cat(preds).squeeze()
            time_labels_tensor = torch.cat(time_labels).squeeze()
            patient_ids_tensor = torch.cat(patient_ids)

            # get a mapping from patient_ids to predictions
            patient_ids_set = set(patient_ids_tensor.numpy())
            prediction_dict = dict()
            for pid, pred, time_label in zip(patient_ids_tensor, preds_tensor, time_labels_tensor):
                prediction_dict[pid.item()] = (pred.numpy(), time_label.numpy())

        elif isinstance(self.state.model_wrapper, TabularWrapper):
            preds_list = predictions[0][0]
            time_labels_list = predictions[0][1]
            patient_ids_list = predictions[0][2]

            patient_ids_set = set(patient_ids_list)
            prediction_dict = dict()
            for pid, pred, time_label in zip(patient_ids_list, preds_list, time_labels_list):
                prediction_dict[pid] = (pred, time_label)

        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported model wrapper: {type(self.state_model_wrapper)}"
            )

        # add the predictions to the patient dataframes
        patient_dfs_scored = []
        count_available_short = 0
        patient_lengths = []
        for patient_df in patient_dfs:
            patient_id = patient_df[PID][0]
            if patient_id not in patient_ids_set:
                continue
            patient_df_copy = patient_df.copy()
            true_length = len(patient_df_copy)

            patient_preds = prediction_dict[patient_id][0]
            patient_time_labels = prediction_dict[patient_id][1]

            available_length = len(patient_preds)
            min_length = min(true_length, available_length)
            patient_lengths.append(min_length)
            if min_length < true_length:
                count_available_short += 1

            patient_preds = patient_preds[:min_length]
            patient_time_labels = patient_time_labels[:min_length]
            patient_time_labels[np.isnan(patient_time_labels)] = -1

            patient_df_copy = patient_df_copy.head(n=min_length)

            patient_df_copy["PredScore"] = patient_preds
            patient_df_copy["TimeLabel"] = patient_time_labels

            patient_dfs_scored.append(patient_df_copy)

        if count_available_short > 0:
            logging.warning(
                f"[{self.__class__.__name__}] {count_available_short} patients too short"
            )
        logging.info(
            f"[{self.__class__.__name__}] Mean patient lengths: {np.mean(patient_lengths):.2f} on split: {split}"
        )

        # Store entire scored dataframes for debugging
        if self.debug:
            df_file = (
                Path(self.state.log_dir)
                / "predictions"
                / f"debug_scored_endpoint_dfs_split_{split}.pkl"
            )
            logging.debug(
                f"[{self.__class__.__name__}] Debug mode: storing entire scored dataframes"
            )
            df_file.parent.mkdir(parents=True, exist_ok=True)
            with open(df_file, "wb") as f:
                pickle.dump(patient_dfs_scored, f, protocol=pickle.HIGHEST_PROTOCOL)

        return patient_dfs_scored

    @staticmethod
    def get_num_alarms_and_detect_distance(patient_scores, thresholds: np.ndarray):
        """
        Compute the precision and recall for the given thresholds and results

        Precision = {True Alarms} / {All Alarms}
        Recall = {Catched Events} / {All Events}
        """
        all_alarms = np.zeros((len(thresholds),), dtype=np.int32)

        distances = np.zeros((len(thresholds),), dtype=np.int32)
        catched_events = np.zeros((len(thresholds),), dtype=np.int32)
        all_events = 0

        for patient_score in patient_scores:

            alarm_scores = patient_score[0]
            all_events += patient_score[1]

            for i in range(len(thresholds)):
                all_alarms[i] += alarm_scores[i, 1]
                catched_events[i] += alarm_scores[i, 2]
                distances[i] += alarm_scores[i, 3]

        mean_distances = distances / (all_events + 1e-6)

        return all_alarms, mean_distances
    

    @staticmethod
    def compute_precision_recall(patient_scores: Any, thresholds: np.ndarray) -> pd.DataFrame:
        """
        Compute the precision and recall for the given thresholds and results

        Precision = {True Alarms} / {All Alarms}
        Recall = {Catched Events} / {All Events}
        """
        true_alarms = np.zeros((len(thresholds),), dtype=np.int32)
        all_alarms = np.zeros((len(thresholds),), dtype=np.int32)

        catched_events = np.zeros((len(thresholds),), dtype=np.int32)
        all_events = 0

        for patient_score in patient_scores:

            alarm_scores = patient_score[0]
            all_events += patient_score[1]

            for i in range(len(thresholds)):
                true_alarms[i] += alarm_scores[i, 0]
                all_alarms[i] += alarm_scores[i, 1]
                catched_events[i] += alarm_scores[i, 2]

        precision = true_alarms / all_alarms
        recall = catched_events / all_events
        alarms_per_patient_mean = all_alarms / len(patient_scores)

        return precision, recall, alarms_per_patient_mean

    @staticmethod
    def plot_precision_recall(
        precision: np.ndarray,
        recall: np.ndarray,
        auc_only: bool = False,
        aux_stats: tuple = None
    ):
        """
        Plot the precision and recall for the given thresholds and results
        """
        precision_mask = np.isnan(precision)
        recall_mask = np.isnan(recall)
        mask = np.logical_or(precision_mask, recall_mask)

        precision = precision[~mask]
        recall = recall[~mask]
        if aux_stats is not None:
            aux_stats = aux_stats[0][~mask], aux_stats[1][~mask] # 1: total num alars, 2: mean alarm distance

        sort_index = np.argsort(recall)
        recall = recall[sort_index]
        precision = precision[sort_index]

        # Assemble Plot DF
        plot_df = pd.DataFrame({"recall": recall, "precision": precision})

        mean_alarm_auc = None
        if aux_stats is not None:
            aux_stats = aux_stats[0][sort_index], aux_stats[1][sort_index]
            plot_df["all_alarms"] = aux_stats[0]
            plot_df["mean_distances"] = aux_stats[1]

        # Dedup values
        plot_df.sort_values(["recall", "precision"], inplace=True)
        plot_df.drop_duplicates(["recall", "precision"], inplace=True)
        plot_df.drop_duplicates("recall", keep="first", inplace=True)
        plot_df.sort_values(["recall", "precision"], inplace=True)

        # Compute AUC
        pr_auc = auc(plot_df["recall"], plot_df["precision"])

        alarm_count_auc = None
        mean_distance_auc = None
        if aux_stats is not None:
            alarm_count_auc = np.trapz(y=plot_df["all_alarms"], x=plot_df["recall"])
            mean_distance_auc = np.trapz(y=plot_df["mean_distances"], x=plot_df["recall"])

        precision = plot_df["precision"].values
        recall = plot_df["recall"].values

        if auc_only:
            return pr_auc, None, None

        if aux_stats is None:

            fig, ax = plt.subplots(figsize=(10, 10))
            pr_plot = sns.lineplot(x=recall, y=precision, ax=ax)
            pr_plot.set(xlabel="Event Recall", ylabel="Alarm Precision")

            pr_plot.set_xlim(0.0, 1.0)
            pr_plot.set_ylim(0.0, 1.0)

            props = dict(boxstyle="round", alpha=0.6)
            auprc_text = f"AuPRC: {pr_auc:.4f}"
            pr_plot.text(
                0.75,
                0.95,
                auprc_text,
                transform=pr_plot.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

        else:

            fig, axs = plt.subplots(1, 3, figsize=(30, 10))
            plt.rcParams.update({"font.size": 12})

            # Set axis limits
            axs[0].set_xlim([0.0, 1.0])
            axs[0].set_ylim([0.0, 1.0])
            
            upper_bound_label = f"Event PR: {pr_auc:.4f} AuPRC"
            upper_bound_plot = sns.lineplot(
                data=plot_df, x="recall", y="precision", label=upper_bound_label,
                ax=axs[0], color="tab:red")
            
            plot_title = "Event Precision-Recall"
            axs[0].set_title(plot_title)
            axs[0].set_xlabel("Event Recall")
            axs[0].set_ylabel("Alarm Precision")
            axs[0].legend(loc="upper right")

            # Plot the number of alarms
            axs[1].set_xlim([0.0, 1.0])
            sns.lineplot(data=plot_df, x="recall", y="all_alarms", ax=axs[1], color="tab:blue")

            axs[1].set_title(f"Number of Alarms, AuC: {alarm_count_auc:.4f}")
            axs[1].set_xlabel("Event Recall")
            axs[1].set_ylabel("Number of Alarms")

            # Plot the mean distance
            axs[2].set_xlim([0.0, 1.0])
            sns.lineplot(data=plot_df, x="recall", y="mean_distances", ax=axs[2], color="tab:blue")

            axs[2].set_title(f"Mean Distance, AuC: {mean_distance_auc:.4f}")
            axs[2].set_xlabel("Event Recall")
            axs[2].set_ylabel("Mean Distance (Time-Steps)")

            plt.suptitle(f"Alarm Model: EEP")

        return pr_auc, fig, (alarm_count_auc, mean_distance_auc)

    def run(self):

        # Plotting settings
        sns.set_style("whitegrid")

        results_dir = Path(self.state.log_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Init WandB
        self.log_wandb, log_wandb = False, False
        if self.state.wandb_project is not None:
            log_wandb = True
            self.log_wandb = log_wandb
            if wandb.run is None:
                logging.info(f"[{self.__class__.__name__}] Initializing wandb run")
                wandb.init(project=self.state.wandb_project, name="Event Evaluation")


        if self.log_wandb:
            hparams_config = gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)
            if self.state.gin_config_files is not None:
                gin_configs = [str(p) for p in self.state.gin_config_files]
                if len(gin_configs) == 1:
                    hparams_config["gin_config"] = gin_configs[0]
                else:
                    hparams_config["gin_config"] = gin_configs

            # remove entries where the value
            # is a gin.config.ConfigurableReference i.e. a Macro
            # all other entries are logged as hyperparameters
            for k, v in hparams_config.items():
                if not isinstance(v, gin.config.ConfigurableReference) and not (
                    isinstance(v, list) and isinstance(v[0], gin.config.ConfigurableReference)
                ):
                    wandb.run.summary[k] = v

        # Load patient endpoint data
        start_time = time.perf_counter()
        self.patient_dfs = self.load_patient_endpoint_data()
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"[{self.__class__.__name__}] Loaded patient data in {elapsed_time:.2f}s")

        # ----------------------------------------
        # Load predictions
        # ----------------------------------------
        patient_dfs_scored = {}
        for split in ["val", "test"]:
            start_time = time.perf_counter()
            patient_dfs_scored[split] = self.load_patient_prediction_data(
                self.patient_dfs, split=split
            )
            elapsed_time = time.perf_counter() - start_time
            logging.info(
                f"[{self.__class__.__name__}] Loaded patient prediction data for {len(patient_dfs_scored[split])} patients"
            )
            logging.info(
                f"[{self.__class__.__name__}] Loaded patient prediction data in {elapsed_time:.2f}s"
            )

        # ----------------------------------------
        # Find/Set silencing time
        # ----------------------------------------
        if self.silence_config.silence_optimize:
            assert not self.detect_full_past_window and not self.drop_events_only_full_window

            best_silence, best_silence_plot = EventEvaluation.optimize_silencing(
                patient_dfs_scored["val"],
                tau_threshs=self.thresholds,
                window_time_steps=self.window_size_min // self.step_size_min,
                reset_time_steps=self.reset_time_min // self.step_size_min,
                silence_times_mins=self.silence_config.silence_optimize_mins,
                merge_time_min=self.merge_time_min,
                one_true_per_event=True,
                mask_reset_time_steps=self.mask_reset_time_steps,
                step_size_min=self.step_size_min,
            )

            selected_silence_time_min = best_silence[0]
            plot_file = Path(self.state.log_dir) / "results" / f"best_silence_time_{split}.png"
            best_silence_plot.figure.savefig(plot_file)
            plt.clf()
            if log_wandb:
                wandb.run.summary[f"val/event_lower_auprc"] = best_silence[1]

            # Check best time on test set and get lower bound Event AuPRC
            best_silence, _ = EventEvaluation.optimize_silencing(
                patient_dfs_scored["test"],
                tau_threshs=self.thresholds,
                window_time_steps=self.window_size_min // self.step_size_min,
                reset_time_steps=self.reset_time_min // self.step_size_min,
                silence_times_mins=[selected_silence_time_min],
                merge_time_min=self.merge_time_min,
                one_true_per_event=True,
            )
            if log_wandb:
                wandb.run.summary[f"test/event_lower_auprc"] = best_silence[1]

        else:
            selected_silence_time_min = self.silence_config.silence_time_min

        # ----------------------------------------
        # Evaluate both splits: test/val
        # ----------------------------------------
        for split in ["val", "test"]:
            logging.info(f"[{self.__class__.__name__}] Evaluating split {split}")

            # ----------------------------------------
            # AuPRC
            # ----------------------------------------
            start_time = time.perf_counter()
            patient_scores = EventEvaluation.compute_batch_patient_scores(
                patient_dfs_scored[split],
                self.thresholds,
                selected_silence_time_min // self.step_size_min,
                self.reset_time_min // self.step_size_min,
                self.window_size_min // self.step_size_min,
                merge_time_steps=self.merge_time_min // self.step_size_min,
                detect_full_past_window=self.detect_full_past_window,
                drop_events_only_full_window=self.drop_events_only_full_window,
                mask_reset_time_steps=self.mask_reset_time_steps,
            )
            elapsed_time = time.perf_counter() - start_time
            logging.info(f"[{self.__class__.__name__}] Computed scores in {elapsed_time:.2f}s")
            self.patient_scores = patient_scores

            # Perform result checks
            assert len(patient_scores) == len(
                patient_dfs_scored[split]
            ), f"scores: {len(patient_scores)}, dfs: {len(patient_dfs_scored[split])}"  # check number of patients
            assert len(patient_scores[0]) == 2  # check tuple dimensions
            assert patient_scores[0][0].shape == (
                len(self.thresholds),
                4,
            )  # check alarm scores dimensions

            start_time = time.perf_counter()
            precisions, recalls, _ = EventEvaluation.compute_precision_recall(
                patient_scores, self.thresholds
            )
            elapsed_time = time.perf_counter() - start_time
            logging.info(f"[{self.__class__.__name__}] Compute PR scores in {elapsed_time:.2f}s")
            self.precisions = precisions
            self.recalls = recalls

            all_alarms, mean_distances = EventEvaluation.get_num_alarms_and_detect_distance(
                patient_scores, self.thresholds
            )

            pr_auc, pr_plot, (alarm_count_auc, mean_distance_auc) = EventEvaluation.plot_precision_recall(precisions, recalls, aux_stats=(all_alarms, mean_distances))
            logging.info(
                f"[{self.__class__.__name__}] PR AUC: {pr_auc:.4f} at {selected_silence_time_min} min silence time"
            )
            logging.info(f"[{self.__class__.__name__}] Alarm Count AuC: {alarm_count_auc:.4f}")
            logging.info(f"[{self.__class__.__name__}] Mean Distance AuC: {mean_distance_auc:.4f}")

            # -------- Store Plot --------
            plot_file = (
                Path(self.state.log_dir)
                / "results"
                / f"pr_plot_split_{split}_silence_{selected_silence_time_min}.png"
            )
            figure_title = f"Precision-Recall Curve: {split} ({selected_silence_time_min} min silencing)"
            try:
                pr_plot.set_title(figure_title)
            except AttributeError as e:
                logging.warning(f"[{self.__class__.__name__}] Could not set title: {e}")
                pr_plot.suptitle(figure_title)

            pr_plot.figure.savefig(plot_file)
            # plt.clf()

            if log_wandb:
                wandb.run.summary[f"{split}/event_silence_time_min"] = selected_silence_time_min
                wandb.run.summary[f"{split}/event_reset_time_min"] = self.reset_time_min
                wandb.run.summary[f"{split}/event_auprc"] = pr_auc
                # wandb.run.summary[f"{split}/event_alarm_auc"] = mean_alarm_auc
                wandb.run.summary[f"{split}/eep/mean_distance_auc"] = mean_distance_auc
                wandb.run.summary[f"{split}/eep/alarm_count_auc"] = alarm_count_auc
                try:
                    wandb.log({f"{split}/event_pr_plot": wandb.Image(pr_plot.figure)})
                except Exception as e:
                    logging.warning(f"[{self.__class__.__name__}] WandB logging failed: {e}")
                    wandb.log({f"{split}/event_pr_plot": wandb.Image(pr_plot)})

            plt.clf()

            if self.store_scores:
                event_score_data = {
                    "split": split,
                    "silence_time_min": selected_silence_time_min,
                    "pr_auc": pr_auc,
                    "precisions": precisions,
                    "recalls": recalls,
                    "patient_scores": patient_scores,
                }
                event_score_file = (
                    Path(self.state.log_dir) / "results" / f"event_scores_split_{split}.pkl"
                )
                with open(event_score_file, "wb") as f:
                    pickle.dump(event_score_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return None
