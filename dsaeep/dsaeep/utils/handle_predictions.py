import logging
import pickle
from pathlib import Path

import gin
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsaeep.pipeline import PipelineState, StatefulPipelineStage
from dsaeep.train.sequence import TabularWrapper


@gin.configurable("HandlePredictions", denylist=["state"])
class HandlePredictions(StatefulPipelineStage):
    """
    Handle Predictions pipeline stage
    """

    name = "Handle Predictions"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        accelerator: str = "cpu",
        batch_size: int = 4,
        use_pred_cache: bool = True,
        split: str = "test",
        feature_columns: list[int] = None,
        keep_only_valid_patients: str = False,
        **kwargs,
    ):
        """Constructor for `HandlePredictions`

        Parameters
        ----------
        state : PipelineState
            the pipeline state
        num_workers : int, optional
            number of workers to use for parallel processing, by default 1
        accelerator : str, optional
            accelerator name, by default "cpu"
        batch_size : int, optional
            batch size, by default 4
        use_pred_cache : bool, optional
            boolean to state whether the predictions are cached on disk, by default True
        split : str, optional
            split name, by default "test"
        feature_columns : list[int], optional
            feature columns to use for tabular wrappers / sklearn models, by default None
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.feature_columns = feature_columns
        self.keep_only_valid_patients = keep_only_valid_patients
        self.use_pred_cache = use_pred_cache
        self.split = split
        self.predictions_path = (
            Path(self.state.log_dir) / "predictions" / f"predictions_split_{self.split}.pkl"
        )

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains predictions and in case of caching on disk check if the file has been cached

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        # check if self.state contains predictions and in case of caching on disk check if the file has been cached
        if self.use_pred_cache:
            return hasattr(self.state, "predictions") & self.predictions_path.exists()
        return hasattr(self.state, "predictions")

    def run(self):
        """Load the predictions into the `PipelineState`.

        Raises
        ------
        ValueError
            Raise error when model wrapper isn't supported
        """

        # if use_pred_cache, check if the directory contains predictions
        if self.use_pred_cache and self.predictions_path.exists():
            logging.warning(
                f"[{self.__class__.__name__}] Loading predictions from {self.predictions_path}"
            )
            with open(self.predictions_path, "rb") as f:
                predictions = pickle.load(f)
            self.state.predictions = predictions
            return

        # Run model and fill self.state.predictions and if use_pred_cache store the predictions on disk
        if isinstance(self.state.model_wrapper, pl.LightningModule):
            assert self.state.dataset_class is not None, "No dataset_class set"
            dataset = self.state.dataset_class(
                self.state.data_path,
                split=self.split,
                return_ids=True,
                keep_only_valid_patients=self.keep_only_valid_patients,
            )
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=1
            )

            trainer = pl.Trainer(
                accelerator=self.accelerator,
                devices=1 if self.accelerator == "gpu" else None,
                logger=False,
            )
            predictions_batch = trainer.predict(self.state.model_wrapper, dataloader)
            preds = []
            patient_ids = []
            for batch in predictions_batch:
                preds.append(batch[0])
                patient_ids.append(batch[2])

            preds_tensor = torch.cat(preds).squeeze()
            patient_ids_tensor = torch.cat(patient_ids)
            _, labels, pids_labels = dataset.get_data_and_labels(
                columns=self.feature_columns, drop_unlabeled=False
            )
            # dictionary mapping pid to (predictions, labels)
            predictions = {}
            for pid, pred in zip(patient_ids_tensor, preds_tensor):
                pid_mask = pids_labels == pid.item()
                label = labels[pid_mask]
                predictions[pid.item()] = (pred.numpy()[: len(label)], label)

        elif isinstance(
            self.state.model_wrapper, TabularWrapper
        ):  # using `TabularWrapper` / Sklearn style model
            assert self.state.dataset_class is not None, "No dataset_class set"
            dataset = self.state.dataset_class(
                self.state.data_path,
                split=self.split,
                return_ids=True,
                keep_only_valid_patients=self.keep_only_valid_patients,
            )

            rep, label, patient_ids = dataset.get_data_and_labels(
                columns=self.feature_columns, drop_unlabeled=False
            )
            preds = self.state.model_wrapper.predict(rep)[:, 1]

            # dictionary mapping pid to (predictions, labels)
            predictions = {}
            logging.info(
                f"[{self.__class__.__name__}] Splitting predictions into patients with time-series"
            )
            for pid in tqdm(np.unique(patient_ids)):
                pid_mask = patient_ids == pid
                predictions[pid] = (preds[pid_mask], label[pid_mask])

        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported model wrapper: {type(self.state.model_wrapper)}"
            )

        if self.use_pred_cache:
            self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.predictions_path, "wb") as f:
                pickle.dump(predictions, f)

        self.state.predictions = predictions
