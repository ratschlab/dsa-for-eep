# ========================================
#
# Pytorch Lightning modules for
# training simple sequence models
#
# ========================================
import logging
import os
from typing import Callable

import gin
import lightgbm
import numpy as np
import pytorch_lightning as pl
import sklearn.base
import torch
import torch.nn as nn
import wandb
from joblib import dump, load
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from wandb.lightgbm import wandb_callback

from dsaeep.models.encoders import SequenceModel
from dsaeep.train.utils import (
    binary_task_masked_select,
    eep_output_transform,
    identity_logit_transform,
    init_wandb,
    sigmoid_binary_output_transform,
    softmax_multi_output_transform,
    surv_task_mask_select,
)


@gin.configurable("SequenceWrapper")
class SequenceWrapper(pl.LightningModule):
    """
    A Pytorch Lightning Wrapper to train
    sequence models (classifiers/regressors)
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = gin.REQUIRED,
        weight_decay: float = 1e-6,
        loss: Callable = gin.REQUIRED,
        label_scaler: Callable = None,
        task: str = "classification/binary",
        l1_reg_emb: float = 0.0,
        print_model: bool = False,
        model_req_mask: bool = False,
        load_batch_data_tuple: bool = False,
    ) -> None:
        """
        Constructor for SequenceWrapper

        Parameters
        ----------
        model : nn.Module
            The model to train
        learning_rate : float
            The learning rate for the optimizer
        weight_decay : float
            The weight decay for the optimizer
        loss : Callable
            The loss function to use
        label_scaler : Callable
            A function to scale the labels
        task : str
            The task type, one of:
                - classification/binary
                - classification/multi
                - regression/single
        l1_reg_emb : float
            The L1 regularization strength for the time-point embedding layer
            usually a single linear layer
        print_model : bool
            Whether to print the model summary
        model_req_mask : bool
            Whether the model requires a mask as input
        load_batch_two_data: bool
            When loading a batch, whether the data element
            is a tuple of (indeces, scaling) (tuple: True) or just features (not a tuple: False)
        """
        super().__init__()

        self.model = model
        if print_model:
            logging.info(f"[{self.__class__.__name__}] Model Architecture:")
            logging.info(self.model)

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.loss = loss
        self.label_scaler = label_scaler
        self.l1_reg_emb = l1_reg_emb
        self.model_req_mask = model_req_mask
        self.load_batch_data_tuple = load_batch_data_tuple
        if self.load_batch_data_tuple:
            logging.info(
                f"[{self.__class__.__name__}] Loading batch as tuple of (indeces, scaling)"
            )

        # set metrics and transforms
        self.output_transform = lambda x: x
        self.logit_transform = identity_logit_transform
        self.set_metrics(task)
        self.smooth_labels = False

        if self.l1_reg_emb > 0.0:
            assert isinstance(self.model, SequenceModel), "Regularizer assumes SequenceModel"
            logging.info(
                f"[{self.__class__.__name__}] Adding L1 regularization to embedding layer with strength {l1_reg_emb}"
            )

    def set_smooth(self, smooth: bool):
        self.smooth_labels = smooth

    def set_metrics(self, task: str):

        if task == "classification/binary":
            self.output_transform = sigmoid_binary_output_transform
            self.logit_transform = binary_task_masked_select
            self.metrics = {
                "train": {
                    "AuPR": BinaryAveragePrecision(validate_args=False),
                    "AuROC": BinaryAUROC(validate_args=False),
                },
                "val": {
                    "AuPR": BinaryAveragePrecision(validate_args=True),
                    "AuROC": BinaryAUROC(validate_args=True),
                },
                "test": {
                    "AuPR": BinaryAveragePrecision(validate_args=True),
                    "AuROC": BinaryAUROC(validate_args=True),
                },
            }

        elif task == "classification/multi":
            self.output_transform = softmax_multi_output_transform
            self.metrics = {}
            raise NotImplementedError(
                f"Multi-Class Classification not yet supported, need to add label scaling"
            )

        elif task == "regression/single":
            self.output_transform = lambda x: x
            self.metrics
            raise NotImplementedError(f"Regression not yet supported, need to add label scaling")

        else:
            raise ValueError(f"Unsupported task type: {task}")

    def l1_regularization(self):
        embedding_layer = self.model.encoder.time_step_embedding
        n_params = sum(
            len(
                p.reshape(
                    -1,
                )
            )
            for p in embedding_layer.parameters()
        )
        return sum(torch.abs(p).sum() for p in embedding_layer.parameters()) / n_params

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        if self.model_req_mask:
            return self.model(x, mask)

        return self.model(x)

    def _get_batch(self, batch):
        if self.load_batch_data_tuple:
            if len(batch) == 5:  # there is also patient ids
                data_indeces, data_scaling, labels, mask, patient_ids = batch
                return (data_indeces, data_scaling), labels, mask, patient_ids
            else:
                data_indeces, data_scaling, labels, mask = batch
                return (data_indeces, data_scaling), labels, mask
        else:
            return batch

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        batch = self._get_batch(batch)
        if len(batch) == 3:
            data, labels, mask = batch
            patient_ids = None
        else:
            data, labels, mask, patient_ids = batch

        if self.smooth_labels:
            labels = labels[..., 0]  # We use hard labels

        logits = self(data, mask)  # calls forward
        preds, _ = self.output_transform((logits, labels))

        return preds, labels, patient_ids

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        # Task specific loss
        loss = self.loss(logits, labels)

        # L1 regularization loss
        if self.l1_reg_emb > 0.0:
            l1_loss = self.l1_reg_emb * self.l1_regularization()
            loss += l1_loss
        else:
            l1_loss = torch.tensor(0.0)

        return loss, l1_loss

    def training_step(self, batch, batch_idx: int):
        data, labels, mask = self._get_batch(batch)
        logits = self(data, mask)  # calls forward

        if self.smooth_labels:
            loss_labels = labels[..., 1]  # We use smooth label for loss
            metric_labels = labels[..., 0]
        else:
            loss_labels = labels
            metric_labels = labels

        logits_flat, labels_flat = self.logit_transform(logits, loss_labels, mask)
        _, metric_labels_flat = self.logit_transform(logits, metric_labels, mask)
        loss, l1_loss = self.compute_loss(logits_flat, labels_flat)
        step_dict = {"loss": loss, "l1_loss": l1_loss.item()}

        preds_flat, labels_trans = self.output_transform((logits_flat, metric_labels_flat))
        for metric in self.metrics["train"].values():
            metric.update(preds_flat.detach().cpu(), labels_trans.detach().cpu())

        return step_dict

    def training_epoch_end(self, outputs) -> None:

        # nan to num as in distributed training some batches may be empty towards end of epoch
        train_loss = np.mean([np.nan_to_num(x["loss"].detach().cpu().numpy()) for x in outputs])
        self.log("train/loss", train_loss, prog_bar=True)  # logger=False)

        if len(outputs) > 0 and "l1_loss" in outputs[0]:
            l1_loss = np.mean([x["l1_loss"] for x in outputs])
            self.log("train/l1_loss", l1_loss, prog_bar=False)

        for name, metric in self.metrics["train"].items():
            metric_val = metric.compute()
            self.log(f"train/{name}", metric_val)
            metric.reset()

    def validation_step(self, batch, batch_idx: int):

        data, labels, mask = self._get_batch(batch)
        logits = self(data, mask)  # calls forward

        if self.smooth_labels:
            loss_labels = labels[..., 1]  # We use smooth label for loss
            metric_labels = labels[..., 0]
        else:
            loss_labels = labels
            metric_labels = labels

        logits_flat, labels_flat = self.logit_transform(logits, loss_labels, mask)
        _, metric_labels_flat = self.logit_transform(logits, metric_labels, mask)
        loss, l1_loss = self.compute_loss(logits_flat, labels_flat)
        step_dict = {"loss": loss, "l1_loss": l1_loss.item()}

        preds_flat, labels_trans = self.output_transform((logits_flat, metric_labels_flat))
        for metric in self.metrics["val"].values():
            metric.update(preds_flat.detach().cpu(), labels_trans.detach().cpu())

        return step_dict

    def validation_epoch_end(self, outputs) -> None:

        # nan to num as in distributed training some batches may be empty towards end of epoch
        val_loss = np.mean([np.nan_to_num(x["loss"].detach().cpu().numpy()) for x in outputs])
        self.log("val/loss", val_loss, prog_bar=True)  # logger=False)

        if len(outputs) > 0 and "l1_loss" in outputs[0]:
            l1_loss = np.mean([x["l1_loss"] for x in outputs])
            self.log("val/l1_loss", l1_loss, prog_bar=False)

        for name, metric in self.metrics["val"].items():
            metric_val = metric.compute()
            self.log(f"val/{name}", metric_val)
            metric.reset()

    def test_step(self, batch, batch_idx: int, dataset_idx: int = 0):

        data, labels, mask = self._get_batch(batch)  # Never smoothed labels
        if self.smooth_labels:
            loss_labels = labels[..., 1]  # We dont use smooth label for loss
            metric_labels = labels[..., 0]
        else:
            loss_labels = labels
            metric_labels = labels
        logits = self(data, mask)  # calls forward

        logits_flat, labels_flat = self.logit_transform(logits, metric_labels, mask)
        loss, l1_loss = self.compute_loss(logits_flat, labels_flat)
        step_dict = {"loss": loss, "l1_loss": l1_loss.item()}

        preds_flat, labels_trans = self.output_transform((logits_flat, labels_flat))
        for metric in self.metrics["test"].values():
            metric.update(preds_flat.detach().cpu(), labels_trans.detach().cpu())

        return step_dict

    def test_epoch_end(self, outputs):
        test_loss = np.mean([x["loss"].item() for x in outputs])
        self.log("test/loss", test_loss, prog_bar=True)  # logger=False)

        if len(outputs) > 0 and "l1_loss" in outputs[0]:
            l1_loss = np.mean([x["l1_loss"] for x in outputs])
            self.log("test/l1_loss", l1_loss, prog_bar=False)

        for name, metric in self.metrics["test"].items():
            metric_val = metric.compute()
            self.log(f"test/{name}", metric_val)
            metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer


@gin.configurable("SurvivalWrapper")
class SurvivalWrapper(SequenceWrapper):
    """
    A Pytorch Lightning Wrapper to train
    Dynamic Survival models (classifiers)
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = gin.REQUIRED,
        weight_decay: float = 1e-6,
        loss: str = "landmarking",
        label_scaler: Callable = None,
        task: str = "classification/binary",
        l1_reg_emb: float = 0.0,
        print_model: bool = False,
        model_req_mask: bool = False,
        load_batch_data_tuple: bool = False,
        pred_horizon: int = 10,
    ) -> None:
        """
        Constructor for SurvivalWrapper

        Parameters
        ----------
        model : nn.Module
            The model to train
        learning_rate : float
            The learning rate for the optimizer
        weight_decay : float
            The weight decay for the optimizer
        loss : Callable
            The loss function to use
        label_scaler : Callable
            A function to scale the labels
        task : str
            The task type, one of:
                - classification/binary
                - classification/multi
                - regression/single
        l1_reg_emb : float
            The L1 regularization strength for the time-point embedding layer
            usually a single linear layer
        print_model : bool
            Whether to print the model summary
        model_req_mask : bool
            Whether the model requires a mask as input
        load_batch_data_tuple: bool
            When loading a batch, whether the data element
            is a tuple of (indeces, scaling) (tuple: True) or just features (not a tuple: False)
        pred_horizon: int
            True horizon in nb of steps
        """
        super().__init__(
            model,
            learning_rate,
            weight_decay,
            loss,
            label_scaler,
            task,
            l1_reg_emb,
            print_model,
            model_req_mask,
            load_batch_data_tuple,
        )

        self.model = model
        if print_model:
            logging.info(f"[{self.__class__.__name__}] Model Architecture:")
            logging.info(self.model)

        self.pred_horizon = pred_horizon
        self.smooth_labels = False
        self.output_transform = lambda x: eep_output_transform(x, self.pred_horizon)

    def set_smooth(self, smooth: bool):
        self.smooth_labels = smooth

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # This is a MLE with landmarking
        if self.loss == "landmarking":
            eep_mask = torch.any((mask != 0), dim=-1)
            loss_sample = (
                nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                * mask
            ).sum(dim=-1)
            loss = torch.masked_select(loss_sample, eep_mask).mean()

        # L1 regularization loss
        if self.l1_reg_emb > 0.0:
            l1_loss = self.l1_reg_emb * self.l1_regularization()
            loss += l1_loss
        else:
            l1_loss = 0.0

        return loss, l1_loss

    def set_pred_horizon(self, pred_horizon: int, verbose: bool = False):
        self.pred_horizon = pred_horizon
        self.output_transform = lambda x: eep_output_transform(x, self.pred_horizon)
        if verbose:
            logging.info(
                f"[{self.__class__.__name__}] Setting prediction horizon to {pred_horizon}"
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        batch = self._get_batch(batch)
        if len(batch) == 3:
            data, labels, mask = batch
            patient_ids = None
        else:
            data, labels, mask, patient_ids = batch

        if self.smooth_labels:
            labels = labels[..., 0]  # We use hard labels
            mask = mask[..., 0]

        logits = self(data, mask)  # calls forward
        preds, labels = self.output_transform((logits, labels))

        return preds, labels, patient_ids

    def training_step(self, batch, batch_idx: int):

        data, labels, mask = self._get_batch(batch)

        if self.smooth_labels:
            loss_labels = labels[..., 1]  # We use smooth label for loss
            metric_labels = labels[..., 0]
            mask = mask[..., 0]

        else:
            loss_labels = labels
            metric_labels = labels

        logits = self(data, mask)  # calls forward
        loss, _ = self.compute_loss(logits, loss_labels, mask)
        step_dict = {"loss": loss}

        preds_eep, labels_eep = self.output_transform((logits, metric_labels))
        preds_eep_flat, labels_eep_flat = surv_task_mask_select(
            preds_eep, labels_eep, metric_labels
        )
        for metric in self.metrics["train"].values():
            metric.update(preds_eep_flat.detach().cpu(), labels_eep_flat.detach().cpu())

        return step_dict

    def validation_step(self, batch, batch_idx: int):

        data, labels, mask = self._get_batch(batch)

        if self.smooth_labels:
            loss_labels = labels[..., 1]  # We use smooth label for loss
            metric_labels = labels[..., 0]
            mask = mask[..., 0]
        else:
            loss_labels = labels
            metric_labels = labels

        logits = self(data, mask)  # calls forward

        loss, _ = self.compute_loss(logits, loss_labels, mask)
        step_dict = {"loss": loss}

        preds_eep, labels_eep = self.output_transform((logits, metric_labels))
        preds_eep_flat, labels_eep_flat = surv_task_mask_select(
            preds_eep, labels_eep, metric_labels
        )
        for metric in self.metrics["val"].values():
            metric.update(preds_eep_flat.detach().cpu(), labels_eep_flat.detach().cpu())

        return step_dict

    def test_step(self, batch, batch_idx: int, dataset_idx: int = 0):

        data, labels, mask = self._get_batch(batch)  # Never smoothed labels

        if self.smooth_labels:
            loss_labels = labels[..., 1]  # We use smooth label for loss
            metric_labels = labels[..., 0]
            mask = mask[..., 0]
        else:
            loss_labels = labels
            metric_labels = labels

        logits = self(data, mask)  # calls forward

        loss, _ = self.compute_loss(logits, loss_labels, mask)
        step_dict = {"loss": loss}

        preds_eep, labels_eep = self.output_transform((logits, metric_labels))
        preds_eep_flat, labels_eep_flat = surv_task_mask_select(
            preds_eep, labels_eep, metric_labels
        )

        for metric in self.metrics["test"].values():
            metric.update(preds_eep_flat.detach().cpu(), labels_eep_flat.detach().cpu())

        return step_dict

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        batch = self._get_batch(batch)
        if len(batch) == 3:
            data, labels, mask = batch
            patient_ids = None
        else:
            data, labels, mask, patient_ids = batch

        if self.smooth_labels:
            labels = labels[..., 0]  # We use hard labels

        logits = self(data, mask)  # calls forward
        preds, labels_eep = self.output_transform((logits, labels))

        return preds, labels_eep, patient_ids
    

class TabularWrapper(object):
    """Dummy class for tabular models"""
    pass