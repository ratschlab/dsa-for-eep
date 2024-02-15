# ===========================================
#
# Event based evaluation for survival models
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
from dsaeep.evaluation.priority_alarm_event import (
    base_silencing_function,
    compute_batch_patient_scores_priority,
    scaling_priority_func,
)
from dsaeep.data.utils import q_exp_param
from dsaeep.pipeline import PipelineState, StatefulPipelineStage
from dsaeep.train.sequence import SurvivalWrapper, TabularWrapper
from dsaeep.train.utils import gin_config_to_readable_dictionary


# ----------------------------------------------------------------------------------------
#
# JiT Evaluation Methods
#
# ----------------------------------------------------------------------------------------
@numba.njit()
def row_wise_any(arr: np.ndarray) -> np.ndarray:
    """
    Compute row wise any, because numba does not
    support axis argument of np.any

    Similar problem mentioned here:
    https://stackoverflow.com/questions/57500001/numba-failure-with-np-mean

    Parameters
    ----------
    arr : np.ndarray[bool, ...]
        The array to compute the row wise any for

    Returns
    -------
    np.ndarray[bool]
        The row wise any array
    """
    res = np.zeros(arr.shape[0], dtype=numba.types.bool_)
    for i in range(arr.shape[0]):
        res[i] = arr[i, :].any()
    return res


@gin.configurable("SurvivalEventEvaluation", denylist=["state"])
class SurvivalEventEvaluation(EventEvaluation):
    """
    Event Evaluation Pipeline Stage
    """

    name = "Survival Event Evaluation"

    def __init__(self, state: PipelineState,
                 survival_pred_horizons_steps: list[int] = [96],
                 fixed_horizon_steps: int = 96,
                 use_lower_bound_estimates: bool = False,
                 advanced_alarm_model_metric: str = "precision",
                 metric_adapt_target_horizon: bool = False,
                 num_sensitivity_thresholds: int = 100,
                 optimize_sigmoid_priority_func: bool = False,
                 optimize_q_exp_priority_func: bool = False,
                 run_q_exp_fixed: bool = False,
                 run_q_exp_flipped_fixed: bool = False,
                 run_q_exp_adaptive: bool = False,
                 q_exp_fixed_h_max: int = 144,
                 q_exp_fixed_gamma: float = 1.0,
                 silencing_func: str = "base"):
        """
        Constructor for the Event Evaluation Pipeline Stage `SurvivalEventEvaluation`

        Parameters
        ----------
        state : PipelineState
            The pipeline state
        survival_pred_horizons_steps : list[int]
            The discrete time steps at which to evaluate the survival model
        fixed_horizon_steps : int
            The fixed horizon step at which to evaluate the `fixed-horizon` alarm model
        use_lower_bound_estimates : bool = False
            Whether to use the lower bound estimates to compute precision/recall estimates
            for the advanced alarm models
        advanced_alarm_model_metric : str = "precision"
            The metric to use for the advanced alarm models treshold optimization
        metric_adapt_target_horizon : bool = False
            For the advanced alarm models, we compute estimates on the validation set. If this
            flag is set, we adapt the evaluation horizon (i.e. the window) to be the same as the
            currently evaluated prediction horizon of the survival model. Otherwise the evaluation horizon
            is the same for all prediction horizons (f.e. 8h)
        """
        super().__init__(state)

        self.survival_pred_horizons_steps = sorted(survival_pred_horizons_steps)
        logging.info(
            f"[{self.__class__.__name__}] Survival prediction horizons: {self.survival_pred_horizons_steps}"
        )

        self.fixed_horizon_steps = fixed_horizon_steps
        logging.info(f"[{self.__class__.__name__}] Fixed horizon steps: {self.fixed_horizon_steps}")

        self.use_lower_bound_estimates = use_lower_bound_estimates
        self.advanced_alarm_model_metric = advanced_alarm_model_metric
        self.metric_adapt_target_horizon = metric_adapt_target_horizon
        logging.info(
            f"[{self.__class__.__name__}] Advanced alarm model metric: {self.advanced_alarm_model_metric}, use lower bound estimates: {self.use_lower_bound_estimates}"
        )

        self.num_sensitivity_thresholds = num_sensitivity_thresholds
        self.sensitivity_thresholds = np.linspace(0, 1, self.num_sensitivity_thresholds)
        logging.info(
            f"[{self.__class__.__name__}] Sensitivity thresholds: {self.num_sensitivity_thresholds}"
        )

        self.optimize_sigmoid_priority_func = optimize_sigmoid_priority_func
        self.optimize_q_exp_priority_func = optimize_q_exp_priority_func
        self.run_q_exp_fixed = run_q_exp_fixed
        self.run_q_exp_flipped_fixed = run_q_exp_flipped_fixed
        self.run_q_exp_adaptive = run_q_exp_adaptive
        self.q_exp_fixed_h_max = q_exp_fixed_h_max
        self.q_exp_fixed_gamma = q_exp_fixed_gamma
        self.silencing_func = base_silencing_function

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

        elif isinstance(self.state.model_wrapper, pl.LightningModule) and isinstance(
            self.state.model_wrapper, SurvivalWrapper
        ):
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

            predictions = []
            for horizon_steps in self.survival_pred_horizons_steps:
                self.state.model_wrapper.set_pred_horizon(pred_horizon=horizon_steps, verbose=True)
                predictions_at_h = trainer.predict(self.state.model_wrapper, dataloader)
                predictions.append(predictions_at_h)

        elif isinstance(
            self.state.model_wrapper, TabularWrapper
        ):  # using `TabularWrapper` / Sklearn style model
            raise NotImplementedError(
                f"[{self.__class__.__name__}] TabularWrapper not implemented yet"
            )

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
            prediction_dicts = []
            for horizon_steps, predictions_at_h in zip(
                self.survival_pred_horizons_steps, predictions
            ):
                preds = []
                time_labels = []
                patient_ids = []
                for batch in predictions_at_h:
                    preds.append(batch[0])
                    time_labels.append(batch[1])
                    patient_ids.append(batch[2])

                preds_tensor = torch.cat(preds).squeeze()
                time_labels_tensor = torch.cat(time_labels).squeeze()
                patient_ids_tensor = torch.cat(patient_ids)

                # get a mapping from patient_ids to predictions
                patient_ids_set = set(patient_ids_tensor.numpy())
                prediction_dict_at_h = dict()
                for pid, pred, time_label in zip(
                    patient_ids_tensor, preds_tensor, time_labels_tensor
                ):
                    prediction_dict_at_h[pid.item()] = (pred.numpy(), time_label.numpy())
                prediction_dicts.append(prediction_dict_at_h)

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

            patient_preds = []
            patient_time_labels = None
            for horizon_steps, prediction_dict in zip(
                self.survival_pred_horizons_steps, prediction_dicts
            ):

                patient_preds_at_h = prediction_dict[patient_id][0]
                patient_preds.append(patient_preds_at_h)

                if horizon_steps == self.fixed_horizon_steps:
                    patient_time_labels = prediction_dict[patient_id][1]

            patient_preds = np.stack(patient_preds, axis=-1)

            available_length = len(patient_preds)
            min_length = min(true_length, available_length)
            patient_lengths.append(min_length)
            if min_length < true_length:
                count_available_short += 1

            patient_preds = patient_preds[:min_length]
            patient_time_labels = patient_time_labels[:min_length]
            patient_time_labels[np.isnan(patient_time_labels)] = -1

            patient_df_copy = patient_df_copy.head(n=min_length)

            patient_df_copy["TimeLabel"] = patient_time_labels
            for i, horizon_steps in enumerate(self.survival_pred_horizons_steps):
                patient_df_copy[f"PredScore_{horizon_steps}"] = patient_preds[:, i]

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

    def prepare_precision_recall(
        self, precision, recall, thresholds, all_alarms, mean_distances, true_alarms
    ) -> pd.DataFrame:

        # Compute mask for NaN values
        precision_mask = np.isnan(precision)
        recall_mask = np.isnan(recall)
        mask = np.logical_or(precision_mask, recall_mask)

        # Remove NaN values
        precision = precision[~mask]
        recall = recall[~mask]
        thresholds = thresholds[~mask]
        all_alarms = all_alarms[~mask]
        mean_distances = mean_distances[~mask]
        true_alarms = true_alarms[~mask]

        # Sort values
        sort_index = np.argsort(recall)
        recall = recall[sort_index]
        precision = precision[sort_index]
        thresholds = thresholds[sort_index]
        all_alarms = all_alarms[sort_index]
        mean_distances = mean_distances[sort_index]
        true_alarms = true_alarms[sort_index]

        # Assemble Plot DF
        pr_df = pd.DataFrame(
            {
                "recall": recall,
                "precision": precision,
                "threshold": thresholds,
                "all_alarms": all_alarms,
                "mean_distances": mean_distances,
                "true_alarms": true_alarms,
            }
        )

        # Dedup values
        pr_df.sort_values(["recall", "precision"], inplace=True)
        pr_df.drop_duplicates(["recall", "precision"], inplace=True)
        pr_df.drop_duplicates("recall", keep="first", inplace=True)
        pr_df.sort_values(["recall", "precision"], inplace=True)

        return pr_df

    @staticmethod
    def get_num_alarms_and_detect_distance(patient_scores, thresholds: np.ndarray):
        """
        Compute the precision and recall for the given thresholds and results

        Precision = {True Alarms} / {All Alarms}
        Recall = {Catched Events} / {All Events}
        """
        true_alarms = np.zeros((len(thresholds),), dtype=np.int32)
        all_alarms = np.zeros((len(thresholds),), dtype=np.int32)

        distances = np.zeros((len(thresholds),), dtype=np.int32)
        catched_events = np.zeros((len(thresholds),), dtype=np.int32)
        all_events = 0

        for patient_score in patient_scores:

            alarm_scores = patient_score[0]
            all_events += patient_score[1]

            for i in range(len(thresholds)):
                true_alarms[i] += alarm_scores[i, 0]
                all_alarms[i] += alarm_scores[i, 1]
                catched_events[i] += alarm_scores[i, 2]
                distances[i] += alarm_scores[i, 3]

        mean_distances = distances / (all_events + 1e-6)

        return all_alarms, mean_distances, true_alarms
    
    @staticmethod
    def compute_gamma_distribution_data(patient_scores, thresholds: np.ndarray):

        if len(patient_scores) <= 0 or len(patient_scores[0]) < 3:
            return None  

        all_gamma_stats = [[] for _ in range(len(thresholds))]
        for patient_score in patient_scores:
            gamma_stats = patient_score[2]
            for i in range(len(thresholds)):
                all_gamma_stats[i].append(gamma_stats[i])

        all_gamma_stats = np.array(all_gamma_stats)
        return all_gamma_stats


    def plot(self,
             precisions_lower, recalls_lower,
             precisions_upper, recalls_upper,
             all_alarms, mean_distances,
             pr_auc_lower, pr_auc_upper,
             gamma_stats,
             true_alarms,
             split: str, save: bool = True,
             alarm_model: str = 'fixed-horizon'):

        logging.info(
            f"[{self.__class__.__name__}] Plotting event precision and recall for split: {split}"
        )
        lower_bound_data = self.prepare_precision_recall(
            precisions_lower,
            recalls_lower,
            self.sensitivity_thresholds,
            np.zeros_like(all_alarms),
            np.zeros_like(mean_distances),
            np.zeros_like(true_alarms),
        )
        upper_bound_data = self.prepare_precision_recall(
            precisions_upper, recalls_upper, self.sensitivity_thresholds,
            all_alarms, mean_distances, true_alarms
        )

        data_dict = {
            "split": split,
            "lower_bound_data": lower_bound_data,
            "upper_bound_data": upper_bound_data,
            "lower_bound_pr_auc": pr_auc_lower,
            "upper_bound_pr_auc": pr_auc_upper,
            "pr_auc_lower": pr_auc_lower,
            "pr_auc_upper": pr_auc_upper,
            "alarm_model": alarm_model,
        }

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        plt.rcParams.update({"font.size": 12})

        # Set axis limits
        axs[0].set_xlim([0.0, 1.0])
        axs[0].set_ylim([0.0, 1.0])

        lower_bound_label = f"Lower Bound: Event PR: {pr_auc_lower:.4f} AuPRC"
        lower_bound_plot = sns.lineplot(
            data=lower_bound_data,
            x="recall",
            y="precision",
            label=lower_bound_label,
            ax=axs[0],
            color="tab:orange",
        )

        upper_bound_label = f"Upper Bound: Event PR: {pr_auc_upper:.4f} AuPRC"
        upper_bound_plot = sns.lineplot(
            data=upper_bound_data,
            x="recall",
            y="precision",
            label=upper_bound_label,
            ax=axs[0],
            color="tab:red",
        )

        plot_title = f"Event Precision-Recall Curve: {split}"
        plot_title += f"\nAlarm Model: {alarm_model}"
        if alarm_model == "multi-horizon":
            plot_title += f" metric: {self.advanced_alarm_model_metric}, lower: {self.use_lower_bound_estimates}"
        plot_title += f"\nSilencing: {self.silence_time_min} min, Reset: {self.reset_time_min} min"
        axs[0].set_title(plot_title)
        axs[0].set_xlabel("Event Recall")
        axs[0].set_ylabel("Alarm Precision")
        axs[0].legend(loc="upper right")

        # Plot the number of alarms
        axs[1].set_xlim([0.0, 1.0])
        sns.lineplot(data=upper_bound_data, x="recall", y="all_alarms", ax=axs[1], color="tab:blue")

        # Compute AuC
        alarm_count_auc = np.trapz(y=upper_bound_data["all_alarms"], x=upper_bound_data["recall"])
        if self.log_wandb:
            wandb.log({f"{split}/{alarm_model}/alarm_count_auc": alarm_count_auc})

        axs[1].set_title(f"Number of Alarms, AuC: {alarm_count_auc:.4f}")
        axs[1].set_xlabel("Event Recall")
        axs[1].set_ylabel("Number of Alarms")

        # Plot the mean distance
        axs[2].set_xlim([0.0, 1.0])
        sns.lineplot(
            data=upper_bound_data, x="recall", y="mean_distances", ax=axs[2], color="tab:blue"
        )

        # Compute AuC for mean distance
        mean_distance_auc = np.trapz(
            y=upper_bound_data["mean_distances"], x=upper_bound_data["recall"]
        )
        if self.log_wandb:
            wandb.log({f"{split}/{alarm_model}/mean_distance_auc": mean_distance_auc})

        axs[2].set_title(f"Mean Distance, AuC: {mean_distance_auc:.4f}")
        axs[2].set_xlabel("Event Recall")
        axs[2].set_ylabel("Mean Distance (Time-Steps)")

        plt.suptitle(f"Alarm Model: {alarm_model}")

        # Save Plot
        if save:
            plot_file = Path(self.save_directory) / f"event_pr_plot_{split}_alarm-{alarm_model}.png"

            fig.savefig(plot_file)
            if self.log_wandb:
                wandb.log({f"{split}/{alarm_model}/event_pr_plot": wandb.Image(fig)})

            plt.clf()

            # Save data dict
            data_dict_file = (
                Path(self.save_directory) / f"event_pr_data_{split}_alarm-{alarm_model}.pkl"
            )
            with open(data_dict_file, "wb") as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


        # Plot the gamma distribution data
        if gamma_stats is None:
            return
        
        # Choose some thresholds to look at the distribution
        gamma_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        fig, axs = plt.subplots(2, 5, figsize=(30, 10))

        for i, threshold in enumerate(gamma_thresholds):
            row = i // 5
            col = i % 5

            # find closest threshold
            threshold_index = np.argmin(np.abs(self.sensitivity_thresholds - threshold))
            matched_threshold = self.sensitivity_thresholds[threshold_index]

            sns.histplot(gamma_stats[threshold_index, :], ax=axs[row, col], bins=50)
            axs[row, col].set_title(f"Threshold: {matched_threshold:.2f}")

        plt.suptitle(f"Gamma Distribution for {split} - {alarm_model}")
        if save:
            plot_file = (
                Path(self.save_directory)
                / f"gamma_distribution_plot_{split}_alarm-{alarm_model}.png"
            )

            fig.savefig(plot_file)
            if self.log_wandb:
                wandb.log({f"{split}/{alarm_model}/gamma_distribution_plot": wandb.Image(fig)})

            plt.clf()

    
    @staticmethod
    def fixed_horizon_scores_transform(patient_dfs_scored, fixed_horizon_steps: int = 96):
        for patient_df in patient_dfs_scored:
            patient_df["PredScore"] = patient_df[f"PredScore_{fixed_horizon_steps}"].values
        return patient_dfs_scored

    def compute_event_scores(
        self, patient_dfs_scored_val, patient_dfs_scored_test, alarm_model: str = "fixed-horizon"
    ):

        logging.info(20 * "-")
        logging.info(f"[{self.__class__.__name__}] Alarm model: {alarm_model}")
        logging.info(20 * "-")

        # Compute Utility Scores or perform optimizations on the validation set
        if alarm_model == "fixed-horizon" or alarm_model == "multi-horizon-area":
            pass

        elif alarm_model == "fixed-priority-func-q-exp":
            logging.info(f"[{self.__class__.__name__}] Running fixed priority function with q_exp")
            logging.info(
                f"[{self.__class__.__name__}] h_max: {self.q_exp_fixed_h_max}, gamma: {self.q_exp_fixed_gamma}"
            )
            weights_arr = np.array(
                [
                    q_exp_param(
                        h, h_true=None, h_max=self.q_exp_fixed_h_max, gamma=self.q_exp_fixed_gamma
                    )
                    for h in self.survival_pred_horizons_steps
                ],
                dtype=np.float32,
            )
            priority_func = partial(scaling_priority_func, weights=weights_arr)
            logging.info(f"[{self.__class__.__name__}] Priority function: {priority_func}")

        elif alarm_model == "fixed-priority-func-q-exp-flipped":
            logging.info(
                f"[{self.__class__.__name__}] Running fixed priority function with *flipped* q_exp"
            )
            logging.info(
                f"[{self.__class__.__name__}] h_max: {self.q_exp_fixed_h_max}, gamma: {self.q_exp_fixed_gamma}"
            )
            weights_arr = np.array(
                [
                    q_exp_param(
                        self.fixed_horizon_steps - h,
                        h_true=None,
                        h_max=self.q_exp_fixed_h_max,
                        gamma=self.q_exp_fixed_gamma,
                    )
                    for h in self.survival_pred_horizons_steps
                ],
                dtype=np.float32,
            )
            weights_arr = 1.0 - weights_arr  # Flip the weights
            priority_func = partial(scaling_priority_func, weights=weights_arr)
            logging.info(f"[{self.__class__.__name__}] Priority function: {priority_func}")
        
        else:
            raise ValueError(f"Unknown alarm model: {alarm_model}")

        # Prepare Alarm Scoring function
        if alarm_model == "fixed-horizon":
            compute_batch_patient_scores_func = partial(
                EventEvaluation.compute_batch_patient_scores,
                tau_threshs=self.sensitivity_thresholds,
                silence_time_steps=self.silence_time_min // self.step_size_min,
                reset_time_steps=self.reset_time_min // self.step_size_min,
                window_time_steps=self.window_size_min // self.step_size_min,
                merge_time_steps=self.merge_time_min // self.step_size_min,
                mask_reset_time_steps=self.mask_reset_time_steps,
                detect_full_past_window=False,
                drop_events_only_full_window=False,
            )

            # Transform scores to fixed horizon scores
            patient_dfs_transform_func = partial(
                SurvivalEventEvaluation.fixed_horizon_scores_transform,
                fixed_horizon_steps=self.fixed_horizon_steps,
            )

        elif (
            alarm_model == "fixed-priority-func-q-exp"
            or alarm_model == "fixed-priority-func-q-exp-flipped"
        ):

            compute_batch_patient_scores_func = partial(
                compute_batch_patient_scores_priority,
                tau_threshs=self.sensitivity_thresholds,
                priority_func=priority_func,
                silencing_func=self.silencing_func,
                survival_pred_horizons_steps=self.survival_pred_horizons_steps,
                silence_time_steps=self.silence_time_min // self.step_size_min,
                reset_time_steps=self.reset_time_min // self.step_size_min,
                window_time_steps=self.window_size_min // self.step_size_min,
                merge_time_steps=self.merge_time_min // self.step_size_min,
                mask_reset_time_steps=self.mask_reset_time_steps,
            )

            patient_dfs_transform_func = lambda x: x

        else:
            raise ValueError(f"Unknown alarm model: {alarm_model}")

        # Compute scores on both validation and test set
        for split, patient_dfs_scored in zip(
            ["test", "val"], [patient_dfs_scored_test, patient_dfs_scored_val]
        ):
            logging.info(
                f"[{self.__class__.__name__}] Computing event scores for split: {split}, alarm model: {alarm_model}"
            )

            # Compute alarm scores for each patient
            patient_scores_upper = compute_batch_patient_scores_func(
                patient_dfs_transform_func(patient_dfs_scored),
                one_true_per_event=False,
            )

            patient_scores_lower = compute_batch_patient_scores_func(
                patient_dfs_transform_func(patient_dfs_scored),
                one_true_per_event=True,
            )

            # Compute precision and recall for each patient
            precisions_upper, recalls_upper, _ = EventEvaluation.compute_precision_recall(
                patient_scores_upper, self.sensitivity_thresholds
            )

            precisions_lower, recalls_lower, _ = EventEvaluation.compute_precision_recall(
                patient_scores_lower, self.sensitivity_thresholds
            )

            # Get number of alarms and distance to event
            all_alarms_upper, mean_distances_upper, true_alarms_upper = SurvivalEventEvaluation.get_num_alarms_and_detect_distance(
                patient_scores_upper, self.sensitivity_thresholds)
            
            # Get gamma distribution data
            gamma_distribution_data = SurvivalEventEvaluation.compute_gamma_distribution_data(
                patient_scores_upper, self.sensitivity_thresholds)

            # Compute AuPRCs
            pr_auc_upper, _, _ = EventEvaluation.plot_precision_recall(
                precisions_upper, recalls_upper, auc_only=True
            )
            pr_auc_lower, _, _ = EventEvaluation.plot_precision_recall(
                precisions_lower, recalls_lower, auc_only=True
            )
            logging.info(
                f"[{self.__class__.__name__}: {alarm_model} : {split}] Event Precision-recall AUC : {pr_auc_lower:.4f} - {pr_auc_upper:.4f}"
            )

            if self.log_wandb:
                wandb.run.summary[f"{split}/{alarm_model}/event_lower_auprc"] = pr_auc_lower
                wandb.run.summary[f"{split}/{alarm_model}/event_upper_auprc"] = pr_auc_upper

            # Plots
            self.plot(precisions_lower, recalls_lower,
                      precisions_upper, recalls_upper,
                      all_alarms_upper, mean_distances_upper,
                      pr_auc_lower, pr_auc_upper,
                      gamma_distribution_data,
                      true_alarms_upper,
                      split=split, alarm_model=alarm_model)

    def run(self):

        # Plotting settings
        sns.set_style("whitegrid")

        results_dir = Path(self.state.log_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        self.save_directory = results_dir

        # Init WandB
        self.log_wandb = False
        if self.state.wandb_project is not None:
            self.log_wandb = True
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

        # ----------------------------------------
        # Load patient endpoint data
        # ----------------------------------------
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
        # Fixed Horizon Alarm Model
        # ----------------------------------------
        self.compute_event_scores(patient_dfs_scored['val'], patient_dfs_scored['test'], alarm_model='fixed-horizon')

        # ----------------------------------------
        # Fixed Priority Function with q_exp Alarm Model
        # Fixed gamma, h_max
        # ----------------------------------------
        if self.run_q_exp_fixed:
            self.compute_event_scores(
                patient_dfs_scored["val"],
                patient_dfs_scored["test"],
                alarm_model="fixed-priority-func-q-exp",
            )

        # ----------------------------------------
        # Fixed Priority Function with q_exp Alarm Model
        # but with flipped function
        # Fixed gamma, h_max
        # ----------------------------------------
        if self.run_q_exp_flipped_fixed:
            self.compute_event_scores(
                patient_dfs_scored["val"],
                patient_dfs_scored["test"],
                alarm_model="fixed-priority-func-q-exp-flipped",
            )

        return None
