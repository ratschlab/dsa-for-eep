# ========================================
#
# Datasets
#
# ========================================
import logging
from pathlib import Path
from typing import Any, Optional, Union

import gin
import numpy as np
import tables
import torch
from scipy.stats import expon, norm
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from dsaeep.data.utils import get_smoothed_labels


@gin.configurable("ICUVariableLengthDataset")
class ICUVariableLengthDataset(Dataset):
    """torch.Dataset built around ICUVariableLengthLoaderTables"""

    def __init__(
        self,
        source_path: Path,
        split: str = "train",
        maxlen: int = -1,
        scale_label: bool = False,
        return_ids: bool = False,
        feature_datatype: str = "float32",
        subsample_train: float = 1.0,
        feature_load_full_ram: bool = False,
        keep_only_valid_patients: bool = True,
    ):
        """
        Constructor for `ICUVariableLengthDataset`
        Ref: https://github.com/ratschlab/HIRID-ICU-Benchmark/blob/master/icu_benchmarks/data/loader.py

        Parameter
        ---------
        source_path: Path
            path to data
        split: str
            split to load in this dataloader from source data
            default: 'train'
        maxlen: int
            Max size of the generated sequence. If -1, takes the max size existing in split.
        scale_label: bool
            Whether or not to train a min_max scaler on labels (For regression stability).
        return_ids: bool
            Whether or not to return the patient id with the sample.
        feature_datatype: str
            Datatype to use for features. Either 'float32' or 'float16'.
            Only used in `get_data_and_labels` method.
        subsample_train: float
            Subsampling rate for training data. Default to 1.0 (no subsampling)
        feature_load_full_ram: bool
            Whether or not to load the full feature matrix to RAM.
        keep_only_valid_patients: bool
            Whether or not to keep only patients with at least one labeled time point.
        """
        logging.info(f"[{self.__class__.__name__}] Loading split: {split}")
        self.keep_only_valid_patients = keep_only_valid_patients
        self.h5_loader = ICUVariableLengthLoaderTables(
            source_path,
            batch_size=1,
            maxlen=maxlen,
            splits=[split],
            keep_only_valid_patients=self.keep_only_valid_patients,
        )
        self.split = split
        self.maxlen = self.h5_loader.maxlen
        self.return_ids = return_ids
        self.feature_load_full_ram = feature_load_full_ram
        self.subsample_train = subsample_train
        if self.subsample_train < 1.0 and self.split == "train":
            logging.warning(
                f"[{self.__class__.__name__}] Subsampling training data to {self.subsample_train}"
            )

        self.scale_label = scale_label
        self.smooth_labels = self.h5_loader.smooth
        if self.scale_label:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.get_labels().reshape(-1, 1))
        else:
            self.scaler = None

        if feature_datatype == "float32":
            self.feature_datatype = np.float32
        elif feature_datatype == "float16":
            self.feature_datatype = np.float16
        else:
            raise ValueError(f"Unknown feature datatype: {feature_datatype}")

    def __len__(self):
        return self.h5_loader.num_samples[self.split]

    def __getitem__(self, idx):
        data, label, pad_mask, patient_id = self.h5_loader.sample(None, self.split, idx)

        # Create torch.Tensor
        if isinstance(data, list):
            data = [torch.from_numpy(d) for d in data]
        else:
            data = torch.from_numpy(data)

        # Scale labels
        if self.scale_label:
            label = self.scaler.transform(label.reshape(-1, 1))[:, 0]

        if self.return_ids:
            return data, torch.from_numpy(label), torch.from_numpy(pad_mask), patient_id

        return data, torch.from_numpy(label), torch.from_numpy(pad_mask)

    def set_scaler(self, scaler: BaseEstimator):
        """
        Sets the scaler for labels in case of regression.

        Parameter
        ---------
        scaler: BaseEstimator
            sklearn scaler instance
        """
        self.scaler = scaler

    def get_labels(self) -> np.ndarray:
        """
        Return all labels of the associated source data
        """
        return self.h5_loader.labels[self.split]

    def get_bias_init(self) -> list[float]:
        """
        Return the weight balance for the split of this dataset
        w.r.t. to the labels

        Returns
        -------
        list:
            Weights for each label.
        """
        if self.h5_loader.surv:
            balance = np.nanmean(self.h5_loader.survival_labels[self.split], axis=0)
            return np.log(balance / (1 - balance))
        else:
            balance = np.nanmean(self.h5_loader.survival_labels[self.split])
            return np.log(balance / (1 - balance))

    def get_balance(self) -> list[float]:
        """
        Return the weight balance for the split of this dataset
        w.r.t. to the labels

        Returns
        -------
        list:
            Weights for each label.
        """
        if self.h5_loader.surv:
            surv_agg = np.nansum(self.h5_loader.survival_labels[self.split], axis=1)
            _, counts = np.unique(surv_agg, return_counts=True)
            return list((1 / counts) * np.sum(counts) / counts.shape[0])
        else:
            labels = self.h5_loader.labels[self.split]
            _, counts = np.unique(labels[np.where(~np.isnan(labels))], return_counts=True)
            return list((1 / counts) * np.sum(counts) / counts.shape[0])

    def get_data_and_labels(
        self, columns=None, drop_unlabeled: bool = True
    ) -> tuple[np.ndarray, ...]:
        """
        Function to return all the data and labels aligned at once.
        We use this function for the ML methods which don't require an iterator.

        Return
        ------
        tuple[np.array, np.array]
            a tuple containing  data points and labels for the split.
        """
        labels = []
        rep = []
        windows = self.h5_loader.patient_windows[self.split][:]
        resampling = self.h5_loader.label_resampling
        logging.info(f"[{self.__class__.__name__}] Gathering the samples for split " + self.split)
        logging.info(
            f"[{self.__class__.__name__}] Converting to data type: {self.feature_datatype}"
        )

        if columns is not None:
            logging.info(f"[{self.__class__.__name__}] Keeping {len(columns)} columns")

        if columns is not None:
            idx_to_keep = np.where(np.isin(self.h5_loader.columns, columns))[0]
            logging.info(f"[{self.__class__.__name__}] Keeping sample columns: {len(idx_to_keep)}")
            all_labels = self.h5_loader.labels[self.split]
            all_data = self.h5_loader.lookup_table[self.split][:, idx_to_keep]
        else:
            all_labels = self.h5_loader.labels[self.split]
            all_data = self.h5_loader.lookup_table[self.split]

        idx_feat_to_keep = None
        if self.h5_loader.feature_table is not None and columns is not None:
            feature_columns_not_duplicates = np.logical_not(
                np.isin(self.h5_loader.features_columns, self.h5_loader.columns)
            )
            bool_feat_to_keep = np.logical_and(
                np.isin(self.h5_loader.features_columns, columns), feature_columns_not_duplicates
            )
            idx_feat_to_keep = np.where(bool_feat_to_keep)[0]
            logging.info(
                f"[{self.__class__.__name__}] Keeping features columns: {len(idx_feat_to_keep)}"
            )
            all_features = self.h5_loader.feature_table[
                self.split
            ]  # [:, idx_feat_to_keep], done below for resource efficiency

        elif self.h5_loader.feature_table is not None:
            all_features = self.h5_loader.feature_table[
                self.split
            ]  # [:, 1:] # done below for resource efficiency

        else:
            logging.warning(f"[{self.__class__.__name__}] No features provided")

        # Load full feature matrix to RAM
        if self.feature_load_full_ram and self.h5_loader.feature_table is not None:
            logging.info(f"[{self.__class__.__name__}] Loading full feature matrix to RAM")
            all_features = all_features[:]

        patient_ids = []
        for i, (start, stop, id_) in enumerate(tqdm(windows)):
            # ignore if not in valid_indexes_samples
            if i not in self.h5_loader.valid_indexes_samples[self.split]:
                continue

            label = all_labels[start:stop][::resampling][: self.maxlen]
            if drop_unlabeled:
                label_mask = ~np.isnan(label)
            else:
                label_mask = np.ones_like(label).astype(bool)

            sample = all_data[start:stop][::resampling][: self.maxlen][label_mask]

            if self.h5_loader.feature_table is not None:
                features = all_features[start:stop][::resampling][: self.maxlen]  # cut to length
                if idx_feat_to_keep is not None:
                    features = features[:, idx_feat_to_keep]
                else:
                    features = features[:, 1:]
                features = features[label_mask]  # remove unlabeled
                sample = np.concatenate((sample, features), axis=-1).astype(self.feature_datatype)

            label = label[label_mask]
            if label.shape[0] > 0:

                if self.subsample_train < 1.0 and self.split == "train":
                    num_rows = label.shape[0]
                    num_rows_to_keep = int(num_rows * self.subsample_train)
                    sub_indeces = np.random.choice(num_rows, num_rows_to_keep, replace=False)
                    sample = sample[sub_indeces]
                    label = label[sub_indeces]

                rep.append(sample)
                labels.append(label)
                patient_ids.append([id_ for _ in range(label.shape[0])])

        rep = np.concatenate(rep, axis=0, dtype=self.feature_datatype)
        labels = np.concatenate(labels)
        patient_ids = np.concatenate(patient_ids, axis=0)
        if self.scaler is not None:
            labels = self.scaler.transform(labels.reshape(-1, 1))[:, 0]

        logging.info(
            f"[{self.__class__.__name__}] Sample matrix (split: {self.split}) shape: {rep.shape}"
        )
        if self.return_ids:
            return rep, labels, patient_ids

        return rep, labels

# ========================================
#
# Table Loader for variable length windows
#
# ========================================
@gin.configurable("ICUVariableLengthLoaderTables")
class ICUVariableLengthLoaderTables(object):
    """Data loader from h5 compressed files with tables to numpy for variable_size windows."""

    def __init__(
        self,
        data_path: Path,
        on_RAM: bool = True,
        shuffle: bool = False,
        batch_size: int = 1,
        splits: list[str] = ["train", "val"],
        maxlen: int = -1,
        task: Optional[Union[int, str]] = 1,
        data_resampling: int = 1,
        label_resampling: int = 1,
        use_feat: bool = False,
        keep_only_valid_patients: bool = True,
        smooth_labels: bool = False,
        surv: bool = False,
        max_horizon: int = -1,
        use_presence_features: bool = False,
        surv_scale: float = 1000,
    ):
        """
        Constructor for `ICUVariableLengthLoaderTables`

        Parameter
        ---------
        data_path: str
            Path to the h5 data file which should have 3 (or 4) subgroups `data`, `labels`, `patient_windows`
            and optionally `features`. Here because arrays have variable length we can't stack them. Instead we
            concatenate them and keep track of the windows in a third variable.
        on_RAM: bool
            Boolean whether to load data on RAM. If you don't have ram capacity set it to False.
        shuffle: bool
            Boolean to decide whether to shuffle data between two epochs when using self.iterate
            method. As we wrap this Loader in a torch Dataset this feature is usually not used.
        batch_size: int
            Integer with size of the batch we return. As we wrap this Loader in a
            torch Dataset this is set to 1.
        splits: list[str]
            list of splits name . Default is ['train', 'val']
        maxlen: int
            Integer with the maximum length of a sequence. If -1 take the maximum length in the data.
        task: Union[int, str]:
            Integer with the index of the task we want to train on in the labels.
            If string we find the matching tring in data_h5['tasks']
        data_resampling: int:
            Number of step at which we want to resample the data.
            Default to 1 (5min HiRID base)
        label_resampling: int
            Number of step at which we want to resample the labels (if they exist).
            Default to 1 (5min HiRID base)
        use_feat: bool
            use special extracted features
        keep_only_valid_patients: bool
            Whether or not to keep only patients with at least one labeled time point.
        smooth_labels: bool
            Whether to smooth labels (Only applies to train and val).
        surv: bool
            Whether to train with survival labels (Only applies to train and val).
        max_horizon: int
            Max horizon considered for survival models. Hence labels in a batch are of size (BS, SEQ_LEN, max_horizon).
        use_presence_features: bool
            Load presence features
        """
        logging.info(f"[{self.__class__.__name__}] Loading data from: {data_path}")

        # We set sampling config
        self.shuffle = shuffle
        if self.shuffle:
            logging.warning(f"[{self.__class__.__name__}] shuffling activated on split: {splits}")

        self.batch_size = batch_size
        self.data_h5 = tables.open_file(data_path, "r").root
        self.splits = splits
        self.maxlen = maxlen
        self.resampling = data_resampling
        self.label_resampling = label_resampling
        self.use_feat = use_feat
        self.use_presence_features = use_presence_features
        self.on_RAM = on_RAM
        self.keep_only_valid_patients = keep_only_valid_patients
        self.smooth = smooth_labels
        self.surv = surv
        self.max_horizon = max_horizon
        # Get data columns
        self.columns = np.array(
            [name.decode("utf-8") for name in self.data_h5["data"]["columns"][:]]
        )

        # Set tasks and label idx
        reindex_label = False
        self.task: Optional[Union[int, str]] = None
        if isinstance(task, str):
            tasks = np.array([name.decode("utf-8") for name in self.data_h5["labels"]["tasks"][:]])
            self.task = task
            if self.task == "Phenotyping_APACHEGroup":
                reindex_label = True

            # Access the task index
            try:
                self.task_idx = np.where(tasks == task)[0][0]
            except IndexError as e:
                raise ValueError(f"Task: {task} not found in {data_path}") from e

        elif isinstance(task, int):
            self.task_idx = task
            tasks = np.array([name.decode("utf-8") for name in self.data_h5["labels"]["tasks"][:]])
            self.task = tasks[self.task_idx]
        else:
            raise ValueError(f"Task: {task} is of unsupported type: {type(task)}")

        possible_event = "_".join(self.task.split("_")[:-1] + ["Event"])
        if possible_event in tasks:
            self.event_idx = np.where(tasks == possible_event)[0][0]
            self.event_name = possible_event
        else:
            self.event_idx = None
            self.event_name = None
            logging.warning(
                f"[{self.__class__.__name__}] Event: {possible_event} for task {self.task} not found in {data_path}"
            )

        logging.info(f"[{self.__class__.__name__}] task: {self.task}, event: {self.event_name}")

        # Processing the data part
        self.lookup_table: Optional[dict[str, Any]] = None
        if self.data_h5.__contains__("data"):
            if on_RAM:  # Faster but comsumes more RAM
                self.lookup_table = {split: self.data_h5["data"][split][:] for split in self.splits}
            else:
                self.lookup_table = {split: self.data_h5["data"][split] for split in self.splits}
        else:
            logging.warning(f"[{self.__class__.__name__}] There is no data provided in {data_path}")

        # Processing the feature part
        self.feature_table: Optional[dict[str, Any]] = None
        self.features_columns: Optional[np.ndarray] = None
        if self.data_h5.__contains__("features") and self.use_feat:
            if on_RAM:  # Faster but comsumes more RAM
                self.feature_table = {
                    split: self.data_h5["features"][split][:] for split in self.splits
                }
            else:
                self.feature_table = {
                    split: self.data_h5["features"][split] for split in self.splits
                }
            self.features_columns = np.array(
                [name.decode("utf-8") for name in self.data_h5["features"]["name_features"][:]]
            )

        # Processing the presence features part
        self.presence_table: Optional[dict[str, Any]] = None
        self.presence_features_columns: Optional[np.ndarray] = None
        if self.use_presence_features:
            assert self.data_h5.__contains__("presence_features")
            if on_RAM:  # Faster but comsumes more RAM
                self.presence_table = {
                    split: self.data_h5["presence_features"][split][:] for split in self.splits
                }
            else:
                self.presence_table = {
                    split: self.data_h5["presence_features"][split] for split in self.splits
                }
            self.presence_features_columns = np.array(
                [
                    name.decode("utf-8")
                    for name in self.data_h5["presence_features"]["name_features"][:]
                ]
            )
            logging.info(
                f"[{self.__class__.__name__}] Using presence features: {len(self.presence_features_columns)}"
            )

        # Processing the label part
        if self.data_h5.__contains__("labels"):
            self.labels = {
                split: self.data_h5["labels"][split][:, self.task_idx] for split in self.splits
            }

            if self.event_idx is not None:
                self.events = {
                    split: self.data_h5["labels"][split][:, self.event_idx].astype(np.float32)
                    for split in self.splits
                }
            else:
                self.events = None

            # We reindex Apache groups to [0,15]
            if reindex_label:
                label_values = np.unique(
                    self.labels[self.splits[0]][np.where(~np.isnan(self.labels[self.splits[0]]))]
                )
                assert len(label_values) == 15

                for split in self.splits:
                    self.labels[split][np.where(~np.isnan(self.labels[split]))] = np.array(
                        list(
                            map(
                                lambda x: np.where(label_values == x)[0][0],
                                self.labels[split][np.where(~np.isnan(self.labels[split]))],
                            )
                        )
                    )
        else:
            logging.error(f"[{self.__class__.__name__}] There is no labels provided in {data_path}")
            self.labels = None

        # Process and load the patient windows
        if self.data_h5.__contains__("patient_windows"):
            # Shape is N_stays x 3. Last dim contains [stay_start, stay_stop, patient_id]
            self.patient_windows = {
                split: self.data_h5["patient_windows"][split][:] for split in self.splits
            }
        else:
            raise Exception(
                f"patient_windows is necessary to split samples, none provided in source data: {data_path}"
            )

        # Some patient might have no labeled time points so we don't consider them in valid samples.
        self.valid_indexes_samples = {
            split: np.array(
                [
                    i
                    for i, k in enumerate(self.patient_windows[split])
                    if np.any(~np.isnan(self.labels[split][k[0] : k[1]]))
                    or not self.keep_only_valid_patients
                ]
            )
            for split in self.splits
        }
        self.num_samples = {split: len(self.valid_indexes_samples[split]) for split in self.splits}
        if not self.keep_only_valid_patients:
            logging.warning(
                f"[{self.__class__.__name__}] Keeping all patients (even if all labels nan): {self.num_samples}"
            )
        else:
            logging.info(f"[{self.__class__.__name__}] num samples: {self.num_samples}")

        # Iterate counters
        self.current_index_training = {"train": 0, "test": 0, "val": 0}

        # Set maximum length
        if self.maxlen == -1:
            seq_lengths = [
                np.max(self.patient_windows[split][:, 1] - self.patient_windows[split][:, 0])
                // self.resampling
                for split in self.splits
            ]
            self.maxlen = np.max(seq_lengths)
        else:
            self.maxlen = self.maxlen // self.resampling
        logging.info(
            f"[{self.__class__.__name__}] max length set to: {self.maxlen} with resampling {self.resampling}"
        )

        if self.surv:
            self.survival_labels = {split: self.surv_labels_split(split) for split in self.splits}

        if self.smooth:
            if self.surv:
                self.surv_scale = surv_scale
                self.rvs = []
                for k in range(1, self.max_horizon + 1):
                    rv = norm(k, k / self.surv_scale)
                    # We center the discretization to have 1 prob when scale is large
                    pmf = rv.cdf(np.arange(1, self.max_horizon + 1) + 0.5) - rv.cdf(
                        np.arange(self.max_horizon) + 0.5
                    )
                    pmf[0] = rv.cdf(1.5)
                    pmf[-1] = 1 - rv.cdf(self.max_horizon - 0.5)
                    self.rvs.append(pmf)
                self.smooth_labels = {
                    split: self.smooth_surv_labels_split(split) for split in self.splits
                }
            else:
                self.smooth_labels = {
                    split: self.smooth_labels_split(split) for split in self.splits
                }

    def smooth_labels_split(self, split):
        patient_window = self.patient_windows[split]
        labels = self.labels[split]
        events = self.events[split]
        smooth_labels = []
        if split == "test":
            return labels
        else:
            for start, stop, id_ in tqdm(patient_window):
                label = np.copy(labels[start:stop])
                event = np.copy(events[start:stop])
                not_labeled = np.where(np.isnan(label))
                if len(not_labeled) > 0:
                    label[not_labeled] = -1
                if not np.all(label == -1):
                    smooth_labels.append(get_smoothed_labels(label, event).astype(np.float32))
                else:
                    smooth_labels.append(label.astype(np.float32))
                if len(not_labeled) > 0:
                    smooth_labels[-1][not_labeled] = np.nan
            smooth_labels = np.concatenate(smooth_labels, axis=0)
            return smooth_labels

    def smooth_surv_labels_split(self, split):
        patient_window = self.patient_windows[split]
        surv_labels = self.survival_labels[split]
        events = self.events[split]
        smooth_labels = []
        if split == "test":
            return surv_labels
        else:
            for start, stop, id_ in tqdm(patient_window):
                label = np.copy(surv_labels[start:stop])
                event = np.copy(events[start:stop])
                smooth_labels.append(self.get_smoothed_surv_labels(label, event))
            smooth_labels = np.concatenate(smooth_labels, axis=0)
            return smooth_labels

    def get_smoothed_surv_labels(self, label, event):

        if np.any(label == 1):
            one_x, one_y = np.where(label == 1)
            smooth_label = np.copy(label)
            fs = np.array([self.rvs[k] for k in one_y])
            smooth_label[one_x] = fs
            smooth_label[np.where(np.isnan(label))] = np.nan
            # ss = np.concatenate([np.ones((fs.shape[0], 1)), 1 - np.cumsum(fs, axis=1)], axis=1)[:, :-1] #shit by 1
            # hs = fs / (1e-10 + ss)
            return smooth_label
            # np_id = idx_durations.cpu().numpy()
            # np_events = events.cpu().numpy()
            #
            # fs = np.array([rvs[k] for k in idx_durations])
            # fs *= np_events.reshape(-1, 1)
            # ss = np.concatenate([np.ones((len(np_id), 1)), 1 - np.cumsum(fs, axis=1)], axis=1)  # Shifted by 1
            # hs = fs / (1e-10 + ss[:, :-1])
            #
            # ss = torch.Tensor(ss).to(events.device)
            # hs = torch.Tensor(hs).to(events.device)
        else:
            return label

    def surv_labels_split(self, split):
        """
        Get survival labels for a split.
        """
        patient_window = self.patient_windows[split]
        surv_labels = []
        for start, stop, id_ in tqdm(patient_window):
            labels = self.get_hazard_labels(start, stop, split)
            surv_labels.append(labels)
        surv_labels = np.concatenate(surv_labels, axis=0)
        return surv_labels

    def get_hazard_labels(self, start, stop, split):
        """
        Gets the hazard labels for
        """
        events = np.copy(self.events[split][start:stop][:: self.resampling])
        early_labels = np.copy(self.labels[split][start:stop][:: self.resampling])

        not_first_step = (1 - events[1:] + events[:-1]) * events[
            1:
        ]  # 1 if previous one is 1 and you are 1
        hazard_labels = np.copy(events[1:])
        hazard_labels[np.where(not_first_step)] = np.nan
        labels = np.copy(
            np.lib.stride_tricks.sliding_window_view(
                np.concatenate([hazard_labels, np.nan * np.ones(self.max_horizon)]),
                self.max_horizon,
                writeable=True,
            )[: len(early_labels)]
        )

        # Handle the case where multiple event happened within max_horizon
        pos_idxs = np.where(labels == 1)
        if np.any(labels == 1):
            post_event_idx_1 = np.concatenate(
                [np.arange(i + 1, self.max_horizon) for t, i in zip(pos_idxs[0], pos_idxs[1])]
            ).astype(int)
            post_event_idx_0 = np.concatenate(
                [np.ones(self.max_horizon - i - 1) * t for t, i in zip(pos_idxs[0], pos_idxs[1])]
            ).astype(int)
            labels[post_event_idx_0, post_event_idx_1] = np.nan
        return labels.astype(np.float32)

    def get_window(
        self, start: int, stop: int, split: str, pad_value: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Windowing function

        Parameter
        ---------
        start: int
            Index of the first element.
        stop: int
            Index of the last element.
        split: str
            Name of the split to get window from.
        pad_value: float
            Value to pad with if stop - start < self.maxlen.

        Returns
        -------
        window: np.array
            Array with data.
        pad_mask: np.array
            1D array with 0 if no labels are provided for the timestep.
        labels: np.array
            1D array with corresponding labels for each timestep.
        """
        # We resample data frequency
        assert self.lookup_table is not None
        window = np.copy(self.lookup_table[split][start:stop][:: self.resampling]).astype(
            np.float32
        )
        labels = np.copy(self.labels[split][start:stop][:: self.resampling])

        if self.feature_table is not None:
            feature = np.copy(self.feature_table[split][start:stop][:: self.resampling])
            window = np.concatenate([window, feature], axis=-1)

        if self.presence_table is not None:
            presence_feature = np.copy(self.presence_table[split][start:stop][:: self.resampling])
            presence_feature = presence_feature.astype(np.float32)[
                :, 1:
            ]  # drop first column, PatientID
            window = np.concatenate([window, presence_feature], axis=-1)

        if self.smooth:
            smooth_labels = np.copy(self.smooth_labels[split][start:stop][:: self.resampling])
            labels = np.stack([labels, smooth_labels], axis=-1)

        label_resampling_mask = np.zeros((stop - start,))
        label_resampling_mask[:: self.label_resampling] = 1.0
        label_resampling_mask = label_resampling_mask[:: self.resampling]
        length_diff = self.maxlen - window.shape[0]
        pad_mask = np.ones((window.shape[0],))

        # If window is shorter than target length
        # we add padding, if its too long we cut it
        if length_diff > 0:
            window = np.concatenate(
                [window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0
            )
            if len(labels.shape) == 1:
                labels_padding = np.ones((length_diff,)) * pad_value
            else:
                labels_padding = np.ones((length_diff, *labels.shape[1:])) * pad_value
            labels = np.concatenate([labels, labels_padding], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros((length_diff,))], axis=0)
            label_resampling_mask = np.concatenate(
                [label_resampling_mask, np.zeros((length_diff,))], axis=0
            )
        elif length_diff < 0:
            window = window[: self.maxlen]
            labels = labels[: self.maxlen]
            pad_mask = pad_mask[: self.maxlen]
            label_resampling_mask = label_resampling_mask[: self.maxlen]

        if len(labels.shape) == 1:
            not_labeled = np.isnan(labels)
            if len(not_labeled) > 0:
                labels[not_labeled] = -1
                pad_mask[not_labeled] = 0
        else:
            not_labeled = np.isnan(labels[:, 0])  # We use the true labels
            if len(not_labeled) > 0:
                labels[not_labeled] = -1
                pad_mask[not_labeled] = 0

        # We resample prediction frequency
        pad_mask = pad_mask * label_resampling_mask
        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        window = window.astype(np.float32)
        return window, labels, pad_mask

    def get_window_surv(self, start, stop, split, pad_value=0.0):
        """Windowing function for survival labels

        Args:
            start (int): Index of the first element.
            stop (int):  Index of the last element.
            split (string): Name of the split to get window from.
            pad_value (float): Value to pad with if stop - start < self.maxlen.

        Returns:
            window: np.array
                Array with data.
            pad_mask: np.array
                1D array with 0 if no labels are provided for the timestep.
            labels: np.array
                1D or 2D array with corresponding HAZARD labels for each timestep.
            early_labels: np.array
            1D or 2D array with corresponding EEP labels for each timestep.
        """
        # We resample data frequency
        window = np.copy(self.lookup_table[split][start:stop][:: self.resampling])
        labels = np.copy(self.survival_labels[split][start:stop][:: self.resampling])
        early_labels = np.copy(self.labels[split][start:stop][:: self.resampling])

        if self.smooth:
            smooth_labels = np.copy(self.smooth_labels[split][start:stop][:: self.resampling])
            labels = np.stack([labels, smooth_labels], axis=-1)

        if self.feature_table is not None:
            feature = np.copy(self.feature_table[split][start:stop][:: self.resampling])
            window = np.concatenate([window, feature], axis=-1)

        label_resampling_mask = np.zeros((stop - start,))
        label_resampling_mask[:: self.label_resampling] = 1.0
        label_resampling_mask = label_resampling_mask[:: self.resampling]
        length_diff = self.maxlen - window.shape[0]
        pad_mask = np.ones_like(labels)

        if length_diff > 0:
            window = np.concatenate(
                [window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0
            )
            labels_padding = (
                np.ones((length_diff, *labels.shape[1:])) * -1
            )  # we pad labels with -1 not pad_value
            labels = np.concatenate([labels, labels_padding], axis=0)
            e_labels_padding = (
                np.ones((length_diff, *early_labels.shape[1:])) * -1
            )  # we pad labels with -1 not pad_value
            early_labels = np.concatenate([early_labels, e_labels_padding], axis=0)
            pad_mask = np.concatenate(
                [pad_mask, np.zeros((length_diff, *labels.shape[1:]))], axis=0
            )
            label_resampling_mask = np.concatenate(
                [label_resampling_mask, np.zeros((length_diff,))], axis=0
            )

        elif length_diff < 0:
            window = window[: self.maxlen]
            labels = labels[: self.maxlen]
            pad_mask = pad_mask[: self.maxlen]
            early_labels = early_labels[: self.maxlen]
            label_resampling_mask = label_resampling_mask[: self.maxlen]  # WE NEVER USE IT

        not_labeled = np.where(np.isnan(labels))
        not_pred = np.where(np.isnan(early_labels))

        if len(not_labeled[0]) > 0:
            labels[not_labeled] = -1
            pad_mask[not_labeled] = 0

        if len(not_pred[0]) > 0:
            labels[not_pred] = -1
            pad_mask[not_pred] = 0

        # We resample prediction frequency
        pad_mask = pad_mask
        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        window = window.astype(np.float32)

        if len(np.where((pad_mask == 1) * (labels == -1))[0]) > 0:
            raise Exception("Mismatch between mask and labeling")
        return window, labels, pad_mask

    def sample(
        self,
        random_state: np.random.RandomState,
        split: str = "train",
        idx_patient: Union[list[int], int, np.ndarray] = None,
    ) -> tuple:
        """
        Function to sample from the data split of choice.

        Parameter
        ---------
        random_state: np.random.RandomState
            np.random.RandomState instance for the idx choice if idx_patient is None.
        split: str
            String representing split to sample from, either 'train', 'val' or 'test'.
        idx_patient: Optional[int]
            Possibility to sample a particular sample given a index.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A sample from the desired distribution as tuple of numpy arrays (sample, label, mask).
        """
        assert split in self.splits

        if idx_patient is None:
            idx_patient = random_state.randint(self.num_samples[split], size=(self.batch_size,))
            state_idx = self.valid_indexes_samples[split][idx_patient]
        else:
            state_idx = self.valid_indexes_samples[split][idx_patient]

        patient_windows = self.patient_windows[split][state_idx]

        X: list[np.ndarray] = []
        y: list[np.ndarray] = []
        pad_masks: list[np.ndarray] = []
        if self.batch_size == 1:
            if self.surv:
                X_window, y_window, pad_masks_window = self.get_window_surv(
                    patient_windows[0], patient_windows[1], split
                )
            else:
                X_window, y_window, pad_masks_window = self.get_window(
                    patient_windows[0], patient_windows[1], split
                )

            id_ = patient_windows[2]
            return X_window, y_window, pad_masks_window, id_

        else:
            for start, stop, id_ in patient_windows:
                window, labels, pad_mask = self.get_window(start, stop, split)
                X.append(window)
                y.append(labels)
                pad_masks.append(pad_mask)

            X_array = np.stack(X, axis=0)
            pad_masks_array = np.stack(pad_masks, axis=0)
            y_array = np.stack(y, axis=0)

            return X_array, y_array, pad_masks_array

    def iterate(self, random_state: np.random.RandomState, split: str = "train") -> tuple:
        """
        Function to iterate over the data split of choice.
        This methods is further wrapped into a generator to build a tf.data.Dataset

        Parameter
        ---------
        random_state: np.random.RandomState
            np.random.RandomState instance for the shuffling.
        split: str
            String representing split to sample from, either 'train', 'val' or 'test'.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A sample corresponding to the current_index from the desired split as tuple of numpy arrays.
        """
        if (self.current_index_training[split] == 0) and self.shuffle:
            random_state.shuffle(self.valid_indexes_samples[split])

        next_idx = list(
            range(
                self.current_index_training[split],
                self.current_index_training[split] + self.batch_size,
            )
        )
        self.current_index_training[split] += self.batch_size

        if self.current_index_training[split] >= self.num_samples[split]:
            n_exceeding_samples = self.current_index_training[split] - self.num_samples[split]
            assert n_exceeding_samples <= self.batch_size
            next_idx = next_idx[: self.batch_size - n_exceeding_samples]
            self.current_index_training[split] = 0

        sample = self.sample(random_state, split, idx_patient=next_idx)
        return sample
