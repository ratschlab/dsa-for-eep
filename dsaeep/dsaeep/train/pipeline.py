# ========================================
#
# Training Pipelines
#
# ========================================
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Union

import gin
import numpy as np
import pytorch_lightning as pl
import sklearn
import tables
import torch
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.feature_selection import SelectFromModel
from torch.utils.data import DataLoader

from dsaeep.data.datasets import ICUVariableLengthDataset
from dsaeep.pipeline import (
    PipelineBase,
    PipelineStage,
    PipelineState,
    StatefulPipelineStage,
)
from dsaeep.train.utils import (
    gin_config_to_readable_dictionary,
    init_wandb,
    load_pickle,
    print_and_save_metrics,
    save_gin_config_file,
)


@gin.configurable("DLTrainPipeline")
class DLTrainPipeline(PipelineBase):
    """
    Preprocessing Pipeline for HiRID
    """

    name = "DL Train Pipeline"

    def __init__(
        self,
        gin_config_files: list[Path],
        log_dir: Path,
        stages: list[Callable] = [],
        num_workers: int = 1,
        random_state: int = 42,
        overwrite: bool = False,
        do_test: bool = True,
        do_train: bool = True,
        wandb_project: str = None,
    ) -> None:
        """
        Constructor for `DLTrainPipeline`

        Parameter
        ---------
        gin_config_files: list[str]
            paths to applicable gin config files
        stages: list[PipelineStage]
            list of additional pipeline stages
        num_workers: int
            number of parallel workers if applicable
        random_stage: int
            random state for reproducable experiments
        """

        # self.model: Optional[torch.nn.Module] = None
        # self.model_wrapper: Optional[pl.LightningModule] = None
        # self.dataset_class: Optional[Callable] = None
        # self.data_path: Optional[Path] = None
        # self.val_dataset: Optional[ICUVariableLengthDataset] = None
        # self.train_dataset: Optional[ICUVariableLengthDataset] = None
        # self.logger: Logger = None

        super().__init__([], num_workers)

        self.state.gin_config_files = gin_config_files
        self.state.gin_bindings = []

        self.state.log_dir = log_dir
        self.state.wandb_project = wandb_project

        setup_train_stage = SetupTrain(self.state, overwrite=overwrite)
        cleanup_stage = CleanupTrain(self.state)

        base_stages: list[PipelineStage] = [setup_train_stage]

        if do_train:
            base_stages.append(TrainWithPL(self.state))

        if do_test:
            base_stages.append(TestModelPL(self.state))

        additional_stages = []
        for stage_constructor in stages:
            stage = stage_constructor(self.state)
            if isinstance(stage, StatefulPipelineStage):
                additional_stages.append(stage)
            else:
                raise ValueError(f"Invalid stage: {stage} must be a `StatefulPipelineStage`")

        final_stages = base_stages + additional_stages + [cleanup_stage]
        self.add_stages(final_stages)

        logging.debug(f"[{self.name}] initialized with stages:")
        for s in self.stages:
            logging.debug(f"\t{s}")


@gin.configurable("MLTrainPipeline")
class MLTrainPipeline(PipelineBase):
    """
    Preprocessing Pipeline for HiRID
    """

    name = "ML Train Pipeline"

    def __init__(
        self,
        gin_config_files: list[Path],
        log_dir: Path,
        stages: list[PipelineStage] = [],
        num_workers: int = 1,
        random_state: int = 42,
        overwrite: bool = False,
        do_test: bool = True,
        do_train: bool = True,
        wandb_project: str = None,
        columns: Union[str, list[str]] = None,
    ) -> None:
        """
        Constructor for `MLTrainPipeline`

        Parameter
        ---------
        gin_config_files: list[str]
            paths to applicable gin config files
        stages: list[PipelineStage]
            list of additional pipeline stages
        num_workers: int
            number of parallel workers if applicable
        random_stage: int
            random state for reproducable experiments
        """
        super().__init__([], num_workers)

        self.state.gin_config_files = gin_config_files
        self.state.gin_bindings = []

        self.state.log_dir = log_dir
        self.state.wandb_project = wandb_project
        self.state.columns = columns

        setup_train_stage = SetupTrainSK(self.state, overwrite=overwrite)
        cleanup_stage = CleanupTrain(self.state)

        base_stages: list[PipelineStage] = [setup_train_stage]

        if do_train:
            base_stages.append(TrainWithSK(self.state))

        if do_test:
            base_stages.append(TestModelSK(self.state))

        additional_stages = []
        for stage_constructor in stages:
            stage = stage_constructor(self.state)
            if isinstance(stage, StatefulPipelineStage):
                additional_stages.append(stage)
            else:
                raise ValueError(f"Invalid stage: {stage} must be a `StatefulPipelineStage`")

        final_stages = base_stages + additional_stages + [cleanup_stage]
        self.add_stages(final_stages)

        logging.debug(f"[{self.name}] initialized with stages:")
        for s in self.stages:
            logging.debug(f"\t{s}")


# ========================================
#
# Pipeline Stages for Training
#
# ========================================
class SetSeeds(PipelineStage):
    """Set random seeds for reproducible experiments"""

    name = "Set Seeds"

    def __init__(self, random_state: int, use_pl: bool = True, reproducible: bool = True) -> None:
        """
        Constructor for `SetSeeds`

        Parameter
        ---------
        random_stage: int
            integer seed for random state
        use_pl: bool
            Use Pytorch Lightning `seed_everything`
        reproducible: bool
            if manually seeding set torch backends and cublas
        """
        super().__init__()
        self.random_state = random_state
        self.use_pl = use_pl
        self.reproducible = reproducible
        self.done = False

    def run(self):
        if self.use_pl:
            pl.seed_everything(self.random_state, workers=True)

        else:
            os.environ["PYTHONHASHSEED"] = str(self.random_state)
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if self.reproducible:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.done = True

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        return self.done


class SetupGin(PipelineStage):
    """Setup and load GIN configurations"""

    name = "Setup GIN"

    def __init__(self, gin_config_files: list[Path] = None, state: PipelineState = None) -> None:
        """
        Constructor for `SetupGIN` pipelien stage

        Parameter
        ---------
        gin_config_files: list[str]
            path to gin config files
        pipeline: PipelineBase
            reference to the pipeline this stage belongs to
        """
        super().__init__()

        self.gin_config_files = gin_config_files
        self.state = state
        self.done = False

        assert_msg = "Only one input of `gin_config_files` or `pipeline` can be given or None"
        assert bool(self.gin_config_files is not None) ^ bool(self.state is not None)

    def run(self) -> Any:
        gin_bindings = []
        if self.state is not None:
            self.gin_config_files = self.state.gin_config_files
            gin_bindings = self.state.gin_bindings

        # Load gin config files
        gin.parse_config_files_and_bindings(self.gin_config_files, gin_bindings)
        self.done = True

    def runnable(self) -> bool:
        if self.state is not None:
            return (
                hasattr(self.state, "gin_config_files")
                and hasattr(self.state, "gin_bindings")
                and len(self.state.gin_config_files) > 0
            )

        else:
            assert self.gin_config_files is not None
            return len(self.gin_config_files) > 0

    def is_done(self) -> bool:
        return self.done


@gin.configurable("SetupTrain")
class SetupTrain(PipelineStage):
    """Setup the pipeline for training"""

    name = "Setup Training"

    def __init__(
        self,
        state: PipelineState,
        model: torch.nn.Module = gin.REQUIRED,
        dataset_class: Callable = gin.REQUIRED,
        data_path: Path = gin.REQUIRED,
        wrapper_class: pl.LightningModule = gin.REQUIRED,
        overwrite: bool = False,
        load_weights: Union[str, Path] = None,
        jit_model: bool = False,
        maxlen: int = None,
        input_dim: int = None,
        batch_size: int = None,
        use_tmp_disk: bool = False,
    ) -> None:
        super().__init__()

        self.state = state
        self.log_dir = state.log_dir
        self.model = model
        self.dataset_class = dataset_class
        self.data_path = data_path
        self.wrapper_class = wrapper_class
        self.overwrite = overwrite
        self.load_weights = Path(load_weights) if isinstance(load_weights, str) else load_weights
        self.jit_model = jit_model
        self.maxlen = maxlen
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.use_tmp_disk = use_tmp_disk

    def run(self) -> Any:

        # Setup log dir
        if os.path.isdir(self.log_dir) and self.load_weights is None:
            test_metric_path = self.log_dir / "test_metrics.pkl"
            test_metrics_exist = os.path.isfile(test_metric_path)
            if self.overwrite or (not test_metrics_exist):
                logging.warning(f"[{self.__class__.__name__}] cleaning log directory")
                logging.warning(
                    f"[{self.__class__.__name__}] overwrite: {self.overwrite}, test metrics exist: {test_metrics_exist}"
                )
                shutil.rmtree(self.log_dir)

        if self.load_weights is None or (
            not os.path.isdir(self.log_dir) and self.load_weights is not None
        ):
            os.makedirs(self.log_dir)
        logging.info(f"[{self.__class__.__name__}] setup log: {self.log_dir}")

        # Copy to temp disk
        if self.use_tmp_disk:
            # get filename from data_path
            filename = self.data_path.name
            tmp_directory = os.environ.get("TMP")
            logging.info(f"[{self.__class__.__name__}] using tmp disk: {tmp_directory}")
            local_copy_path = Path(tmp_directory) / filename

            # copy data to tmp disk
            if not local_copy_path.exists():
                logging.info(f"[{self.__class__.__name__}] copying data to tmp disk")
                shutil.copy(self.data_path, local_copy_path)

            self.data_path = local_copy_path  # set data path to tmp disk location

        # Init. datasets for training
        self.state.dataset_class = self.dataset_class
        self.state.data_path = self.data_path
        self.state.train_dataset = self.dataset_class(self.data_path, split="train")
        self.state.val_dataset = self.dataset_class(self.data_path, split="val")
        logging.info(f"[{self.__class__.__name__}] loaded datasets [train, val]")

        # We set the label scaler
        assert self.state.val_dataset is not None and self.state.train_dataset is not None

        if hasattr(self.state.train_dataset, "scaler"):
            self.state.val_dataset.set_scaler(self.state.train_dataset.scaler)

        # Set model
        self.state.model = self.model
        model = self.model

        # Torch Settings
        torch.set_float32_matmul_precision("high")
        if self.jit_model:
            logging.warning(
                f"[{self.__class__.__name__}] torch jit trace not yet correctly implemented"
            )
            example_input = torch.rand((self.batch_size, self.maxlen, self.input_dim))
            model = torch.jit.trace(self.state.model, example_input)

        if hasattr(self.state.train_dataset, "scaler"):
            label_scaler = self.state.train_dataset.scaler
            self.state.model_wrapper = self.wrapper_class(model=model, label_scaler=label_scaler)
        else:
            self.state.model_wrapper = self.wrapper_class(model=model)

        logging.info(f"[{self.__class__.__name__}] loaded model and wrapper")

        # Save active config
        save_gin_config_file(self.log_dir)

        # Load model weights if applicable
        if self.load_weights is not None:
            weights_path = None
            if self.load_weights.is_file() and self.load_weights.suffix == ".ckpt":
                weights_path = self.load_weights
            elif self.load_weights.is_dir():
                file_regex = "*.ckpt"
                matched_files = list(self.load_weights.rglob(file_regex))
                assert (
                    len(matched_files) == 1
                ), f"{len(matched_files)} checkpoint ({file_regex}) files matched here: {self.load_weights}"
                weights_path = matched_files[0]
            else:
                raise ValueError(f"Cannot load weights with: {self.load_weights}")

            logging.info(f"[{self.__class__.__name__}] Load model: {weights_path}")
            model_state_dict = torch.load(weights_path)["state_dict"]

            assert self.state.model_wrapper is not None
            missing_keys, unexpected_keys = self.state.model_wrapper.load_state_dict(
                model_state_dict
            )
            logging.info(f"[{self.__class__.__name__}] Missing keys: {missing_keys}")
            logging.info(f"[{self.__class__.__name__}] Unexpected keys: {unexpected_keys}")

    def runnable(self) -> bool:

        if os.path.isdir(self.log_dir) and self.load_weights is None:
            test_metric_path = self.log_dir / "test_metrics.pkl"
            test_metrics_exist = os.path.isfile(test_metric_path)

            if not self.overwrite and test_metrics_exist:
                logging.error(
                    f"[{self.__class__.__name__}] cannot run {test_metric_path} exists: {test_metrics_exist}"
                )
                logging.error(f"[{self.__class__.__name__}] and overwrite: {self.overwrite}")
                return False

        jit_applicable = True
        if self.jit_model:
            jit_applicable = (
                self.maxlen is not None
                and self.input_dim is not None
                and self.batch_size is not None
            )

        return jit_applicable

    def is_done(self) -> bool:
        return False


@gin.configurable("TrainWithPL")
class TrainWithPL(PipelineStage):
    """Train a model using PyTorch Lightning"""

    name = "Train with PL Model"

    def __init__(
        self,
        state: PipelineState,
        batch_size: int = gin.REQUIRED,
        max_epochs: int = gin.REQUIRED,
        num_workers: int = 1,
        autograd_anomaly: bool = False,
        pin_memory: bool = False,
        class_weights: Union[str, list[float]] = None,
        accelerator: str = "cpu",
        num_devices: Union[str, int] = "auto",
        early_stopping_patience: int = 4,
        accumulate_grad_batches: int = 1,
        strategy: str = None,
        grad_clip_val: float = None,
        imbalance_bias_init: bool = False,
    ) -> None:
        super().__init__()

        self.state = state
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        self.autograd_anomaly = autograd_anomaly
        self.pin_memory = pin_memory
        self.class_weights = class_weights
        self.accelerator = accelerator
        self.early_stopping_patience = early_stopping_patience
        self.accumulate_grad_batches = accumulate_grad_batches
        self.num_devices = num_devices
        self.strategy = strategy
        self.grad_clip_val = grad_clip_val
        self.imbalance_bias_init = imbalance_bias_init

    def run(self) -> Any:
        torch.autograd.set_detect_anomaly(self.autograd_anomaly)

        assert self.state.train_dataset is not None
        if not self.state.train_dataset.h5_loader.on_RAM:
            logging.warning(f"[{self.__class__.__name__}] data not loaded to RAM for training")

        logging.info(f"[{self.__class__.__name__}] using {self.num_workers} workers")
        assert self.state.train_dataset is not None and self.state.val_dataset is not None

        train_loader = DataLoader(
            self.state.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=None,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.state.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=None,
        )

        if isinstance(self.class_weights, list):
            self.class_weights = torch.FloatTensor(self.class_weights)
        elif isinstance(self.class_weights, str) and self.class_weights == "balanced":
            self.class_weights = torch.FloatTensor(self.state.train_dataset.get_balance())

        if self.state.wandb_project is not None:
            logger = WandbLogger(project=self.state.wandb_project, save_dir=self.state.log_dir)
            logging.info(f"[{self.__class__.__name__}] logging to WandB: {logger.root_dir}")
            logging.info(f"[{self.__class__.__name__}] \t name: {logger._name}, {logger._id}")
        else:
            logger = TensorBoardLogger(self.state.log_dir, name=f"tb")
            logging.info(f"[{self.__class__.__name__}] logging to Tensorboard: {logger.root_dir}")

        self.state.logger = logger
        if self.state.logger is not None:
            hparams_config = gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)

            if self.state.gin_config_files is not None:
                gin_configs = [str(p) for p in self.state.gin_config_files]
                if len(gin_configs) == 1:
                    hparams_config["gin_config"] = gin_configs[0]
                else:
                    hparams_config["gin_config"] = gin_configs

            self.state.logger.log_hyperparams(hparams_config)

        # Early Stopping
        early_stopping = EarlyStopping(
            monitor=f"val/loss", mode="min", patience=self.early_stopping_patience
        )

        # Checkpointing
        checkpointing = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1)

        if self.accumulate_grad_batches > 1:
            logging.info(
                f"[{self.__class__.__name__}] using gradient accumulation: {self.accumulate_grad_batches}"
            )

        if self.grad_clip_val is not None:
            logging.info(
                f"[{self.__class__.__name__}] using gradient clipping: {self.grad_clip_val}"
            )

        logging.info(f"[{self.__class__.__name__}] using '{self.num_devices}' GPUs")
        logging.info(f"[{self.__class__.__name__}] strategy: {self.strategy}")

        # train model with PL
        trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.num_devices if self.accelerator == "gpu" else None,
            max_epochs=self.max_epochs,
            logger=logger,
            callbacks=[early_stopping, checkpointing],
            # enable_checkpointing=True,
            accumulate_grad_batches=self.accumulate_grad_batches,
            strategy=self.strategy,
            gradient_clip_val=self.grad_clip_val,
        )

        assert self.state.model_wrapper is not None

        if hasattr(self.state.model_wrapper, "set_smooth"):
            self.state.model_wrapper.set_smooth(self.state.train_dataset.smooth_labels)

        if self.imbalance_bias_init:
            bias_init = torch.FloatTensor(self.state.train_dataset.get_bias_init())
            self.state.model_wrapper.model.update_bias_logit(bias_init)

        trainer.fit(self.state.model_wrapper, train_loader, val_loader)

        # Load best module
        if self.state.model_wrapper.current_epoch > 1:
            logging.info(
                f"[{self.__class__.__name__}] Load best validation loss model: {checkpointing.best_model_path}"
            )
            logging.info(f"[{self.__class__.__name__}] \t score: {checkpointing.best_model_score}")
            best_model_state_dict = torch.load(checkpointing.best_model_path)["state_dict"]

            missing_keys, unexpected_keys = self.state.model_wrapper.load_state_dict(
                best_model_state_dict
            )
            logging.info(f"[{self.__class__.__name__}] Missing keys: {missing_keys}")
            logging.info(f"[{self.__class__.__name__}] Unexpected keys: {unexpected_keys}")

    def runnable(self) -> bool:

        # Check dataset availability
        dataset_provided = (
            hasattr(self.state, "train_dataset")
            and self.state.train_dataset is not None
            and hasattr(self.state, "val_dataset")
            and self.state.val_dataset is not None
        )

        model_provided = hasattr(self.state, "model") and self.state.model is not None

        log_dir_provided = hasattr(self.state, "log_dir") and self.state.log_dir is not None

        return dataset_provided and model_provided and log_dir_provided

    def is_done(self) -> bool:
        return False


@gin.configurable("TestModelPL")
class TestModelPL(PipelineStage):
    """Test a model using PyTorch Lightning API"""

    name = "Test with PL API"

    def __init__(
        self,
        state: PipelineState,
        accelerator: str = "cpu",
        batch_size: int = 8,
        num_workers: int = 1,
        test_split: str = "test",
    ) -> None:
        super().__init__()
        self.state = state
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.done = False
        self.num_workers = num_workers
        self.test_split = test_split

    def run(self) -> Any:
        # Assumes we have executed `runnable` to check
        # either parent pipeline or arguments provide data
        assert self.state.dataset_class is not None
        test_dataset: ICUVariableLengthDataset = self.state.dataset_class(
            self.state.data_path, split=self.test_split
        )

        if hasattr(self.state.train_dataset, "scaler"):
            test_dataset.set_scaler(self.state.train_dataset.scaler)  # type: ignore

        assert self.state.val_dataset is not None
        val_dataloader = DataLoader(
            self.state.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        assert self.state.train_dataset is not None

        if hasattr(self.state.train_dataset, "get_balance"):
            weight = self.state.train_dataset.get_balance()

        trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=1 if self.accelerator == "gpu" else None,
            logger=False,
        )

        val_metrics = trainer.test(
            model=self.state.model_wrapper, dataloaders=val_dataloader, verbose=False
        )
        print_and_save_metrics(
            self.state.log_dir, val_metrics[0], prefix="val", print_metrics=True, strip_keys=5
        )

        test_metrics = trainer.test(
            model=self.state.model_wrapper, dataloaders=test_dataloader, verbose=False
        )
        print_and_save_metrics(
            self.state.log_dir, test_metrics[0], prefix=self.test_split, print_metrics=True, strip_keys=5
        )

        del test_dataset.h5_loader.lookup_table
        self.done = True

    def runnable(self) -> bool:
        # Check if we can load the dataset given parent information
        data_provided = (
            hasattr(self.state, "dataset_class")
            and self.state.dataset_class is not None
            and hasattr(self.state, "data_path")
            and self.state.data_path is not None
            and hasattr(self.state, "train_dataset")
        )

        model_provided = hasattr(self.state, "model") and self.state.model is not None

        log_provided = hasattr(self.state, "log_dir") and self.state.log_dir is not None

        return data_provided and model_provided and log_provided

    def is_done(self) -> bool:
        return self.done


class CleanupTrain(PipelineStage):
    """Cleanup after a training run"""

    name = "Cleanup after Train"

    def __init__(self, state: PipelineState) -> None:
        super().__init__()
        self.state = state
        self.done = False

    def run(self) -> Any:

        # Free H5
        if hasattr(self.state, "train_dataset") and self.state.train_dataset is not None:
            del self.state.train_dataset.h5_loader.lookup_table
        if hasattr(self.state, "val_dataset") and self.state.val_dataset is not None:
            del self.state.val_dataset.h5_loader.lookup_table

        # Close open tables/h5 files
        tables.file._open_files.close_all()

        # Save Final operative Gin
        save_gin_config_file(self.state.log_dir, filename="final_config.gin")

        # log hyperparameter and final metrics
        if self.state.logger is not None:

            val_metrics = {}
            test_metrics = {}

            val_path = Path(self.state.log_dir) / "val_metrics.pkl"
            test_path = Path(self.state.log_dir) / "test_metrics.pkl"

            if val_path.exists():
                val_metrics = load_pickle(val_path)
                val_metrics = {f"final/val/{k}": v for k, v in val_metrics.items()}
                self.state.logger.log_metrics(val_metrics)

            if test_path.exists():
                test_metrics = load_pickle(test_path)
                test_metrics = {f"final/test/{k}": v for k, v in test_metrics.items()}
                self.state.logger.log_metrics(test_metrics)

        # Clear Gin
        gin.clear_config()

        # Cleanup WandB
        if self.state.wandb_project is not None:
            wandb.finish()

        self.done = True

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        return self.done


@gin.configurable("SetupTrainSK")
class SetupTrainSK(PipelineStage):
    """Setup the pipeline for training"""

    name = "Setup Training for sci-kit models"

    def __init__(
        self,
        state: PipelineState,
        model: torch.nn.Module = gin.REQUIRED,
        dataset_class: Callable = gin.REQUIRED,
        data_path: Path = gin.REQUIRED,
        wrapper_class: Callable = gin.REQUIRED,
        overwrite: bool = False,
        load_weights: Union[str, Path] = None,
        input_dim: int = None,
    ) -> None:
        super().__init__()

        self.state = state
        self.log_dir = state.log_dir
        self.model = model
        self.dataset_class = dataset_class
        self.data_path = data_path
        self.wrapper_class = wrapper_class
        self.overwrite = overwrite
        self.load_weights = Path(load_weights) if isinstance(load_weights, str) else load_weights
        self.input_dim = input_dim

    def run(self) -> Any:

        # Setup log dir
        if os.path.isdir(self.log_dir) and self.load_weights is None:
            test_metric_path = self.log_dir / "test_metrics.pkl"
            test_metrics_exist = os.path.isfile(test_metric_path)
            if self.overwrite or (not test_metrics_exist):
                logging.warning(f"[{self.__class__.__name__}] cleaning log directory")
                logging.warning(
                    f"[{self.__class__.__name__}] overwrite: {self.overwrite}, test metrics exist: {test_metrics_exist}"
                )
                shutil.rmtree(self.log_dir)

        if self.load_weights is None:
            os.makedirs(self.log_dir)
        logging.info(f"[{self.__class__.__name__}] setup log: {self.log_dir}")

        # Init. datasets for training
        self.state.dataset_class = self.dataset_class
        self.state.data_path = self.data_path
        self.state.train_dataset = self.dataset_class(self.data_path, split="train")
        self.state.val_dataset = self.dataset_class(self.data_path, split="val")
        logging.info(f"[{self.__class__.__name__}] loaded datasets [train, val]")

        # We set the label scaler
        assert self.state.val_dataset is not None and self.state.train_dataset is not None

        if hasattr(self.state.train_dataset, "scaler"):
            self.state.val_dataset.set_scaler(self.state.train_dataset.scaler)

        # Set model
        self.state.model = self.model
        model = self.model

        if hasattr(self.state.train_dataset, "scaler"):
            label_scaler = self.state.train_dataset.scaler
            self.state.model_wrapper = self.wrapper_class(model=model, label_scaler=label_scaler)
        else:
            self.state.model_wrapper = self.wrapper_class(model=model)

        logging.info(f"[{self.__class__.__name__}] loaded model and wrapper")

        # Save active config
        save_gin_config_file(self.log_dir)

        # Load model weights if applicable
        if self.load_weights is not None:
            if self.load_weights.is_file() and self.load_weights.suffix == ".joblib":
                weights_path = self.load_weights
            elif self.load_weights.is_dir():
                file_regex = "*.joblib"
                matched_files = list(self.load_weights.rglob(file_regex))
                assert (
                    len(matched_files) == 1
                ), f"{len(matched_files)} checkpoint ({file_regex}) files matched here: {self.load_weights}"
                weights_path = matched_files[0]
            else:
                raise ValueError(f"Cannot load weights with: {self.load_weights}")
            logging.info(f"[{self.__class__.__name__}] Load model: {weights_path}")
            assert self.state.model_wrapper is not None
            self.state.model_wrapper.load_model(weights_path)

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        return False


@gin.configurable("VariableSelectionSK")
class VariableSelectionSK(StatefulPipelineStage):
    """Perform variable selection using sklearn"""

    name = "Variable Selection SK"

    def __init__(
        self,
        state: PipelineState,
        top_k: int = gin.REQUIRED,
        importance_model: Callable = gin.REQUIRED,
        num_workers: int = 1,
    ) -> None:
        """
        Constructor for `VariableSelectionSK`

        Parameter
        ---------
        state: PipelineState
            reference to the pipeline state
        top_k: int
            number of features to select
        importance_model: Callable
            sklearn model to use for feature importance
        num_workers: int
            -
        """
        super().__init__(state=state, num_workers=num_workers)

        self.state = state
        self.num_workers = num_workers

        self.top_k = top_k
        self.importance_model = importance_model

    def run(self) -> Any:

        # Check the training dataset is loaded
        assert self.state.train_dataset is not None

        # Load the training data matrix
        train_rep, train_label = self.state.train_dataset.get_data_and_labels()

        # Init the importance model
        importance_model = self.importance_model()
        logging.info(
            f"[{self.__class__.__name__}] extract top {self.top_k} features using {importance_model.__class__.__name__}"
        )

        # Perform variable selection
        selector = SelectFromModel(
            estimator=importance_model, max_features=self.top_k, prefit=False
        ).fit(train_rep, train_label)

        # Get the selected features
        selected_features = selector.get_support(indices=False)

        # Get column names
        all_column_names = self.state.train_dataset.h5_loader.columns
        if len(selected_features) > len(
            all_column_names
        ):  # need to load additional extracted features
            all_column_names = np.concatenate(
                [all_column_names, self.state.train_dataset.h5_loader.features_columns[1:]]
            )

        # Extract the selected features
        selected_column_names = all_column_names[selected_features]
        logging.info(f"[{self.__class__.__name__}] selected {len(selected_column_names)} features")

        # Store the selected features list to a text file
        # with comma separated values
        selected_features_path = self.state.log_dir / "selected_features.txt"
        logging.info(
            f"[{self.__class__.__name__}] saving selected features to {selected_features_path}"
        )
        with open(selected_features_path, "w") as f:
            f.write(",".join(selected_column_names))

        selected_features_path_string = self.state.log_dir / "selected_features_string.txt"
        with open(selected_features_path_string, "w") as f:
            f.write(",".join(map(lambda x: f"'{x}'", selected_column_names)))

    def runnable(self) -> bool:

        # Check dataset availability
        dataset_provided = (
            hasattr(self.state, "train_dataset") and self.state.train_dataset is not None
        )

        log_dir_provided = hasattr(self.state, "log_dir") and self.state.log_dir is not None

        return dataset_provided and log_dir_provided

    def is_done(self) -> bool:

        feature_file_exists = False
        if hasattr(self.state, "log_dir") and self.state.log_dir is not None:
            selected_features_path = self.state.log_dir / "selected_features.txt"
            feature_file_exists = selected_features_path.exists()

        return feature_file_exists


@gin.configurable("TrainWithSK")
class TrainWithSK(PipelineStage):
    """Train a model using SKlearn"""

    name = "Train with SK Model"

    def __init__(
        self,
        state: PipelineState,
        max_iter: int = gin.REQUIRED,
        num_workers: int = 1,
        class_weights: Union[str, list[float]] = None,
        accelerator: str = "cpu",
        early_stopping_patience: int = 4,
    ) -> None:
        super().__init__()

        self.state = state
        self.max_iter = max_iter
        self.num_workers = num_workers
        self.class_weights = class_weights
        self.accelerator = accelerator
        self.early_stopping_patience = early_stopping_patience
        self.columns = self.state.columns

    def run(self) -> Any:

        # Log all hyperparameters (gin) to WandB
        if self.state.wandb_project is not None:
            logging.info(
                f"[{self.__class__.__name__}] logging to WandB: {self.state.wandb_project}"
            )
            init_wandb(self.state.wandb_project)
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

        assert self.state.train_dataset is not None
        if not self.state.train_dataset.h5_loader.on_RAM:
            logging.warning(f"[{self.__class__.__name__}] data not loaded to RAM for training")

        logging.info(f"[{self.__class__.__name__}] using {self.num_workers} workers")
        assert self.state.train_dataset is not None and self.state.val_dataset is not None
        model = self.state.model_wrapper.model
        if "class_weight" in model.get_params().keys():  # Set class weights
            model.set_params(class_weight=self.class_weights)
        else:
            raise Exception(
                "Parameter class_weight doesn't exist for classifier {}".format(
                    self.__class__.__name__
                )
            )

        if "n_estimators" in model.get_params().keys():  # Set class weights
            model.set_params(n_estimators=self.max_iter)
        else:
            raise Exception(
                "Parameter n_estimators doesn't exist for classifier {}".format(
                    self.__class__.__name__
                )
            )
        if "n_jobs" in model.get_params().keys():  # Set class weights
            model.set_params(n_jobs=self.num_workers)
        else:
            raise Exception(
                "Parameter n_jobs doesn't exist for classifier {}".format(self.__class__.__name__)
            )

        train_rep, train_label = self.state.train_dataset.get_data_and_labels(columns=self.columns)
        val_rep, val_label = self.state.val_dataset.get_data_and_labels(columns=self.columns)

        assert self.state.model_wrapper is not None
        self.state.model_wrapper.train(
            train_rep,
            train_label,
            eval_set=(val_rep, val_label),
            early_stopping_rounds=self.early_stopping_patience,
            wandb_project=self.state.wandb_project,
            eval_metric=["binary", "auc", "average_precision"],
        )

        trained_model = self.state.model_wrapper.model

        # Load best module
        if self.state.model_wrapper.trained:
            logging.info(f"[{self.__class__.__name__}] Best validation loss model already loaded")
            best_score = list(trained_model.best_score_["valid_0"].values())[0]
            logging.info(
                f"[{self.__class__.__name__}] \t score: {best_score}, iteration {trained_model.best_iteration_}"
            )
            self.state.model_wrapper.save_model(self.state.log_dir)
        else:
            raise Exception("Model should be trained !")

    def runnable(self) -> bool:

        # Check dataset availability
        dataset_provided = (
            hasattr(self.state, "train_dataset")
            and self.state.train_dataset is not None
            and hasattr(self.state, "val_dataset")
            and self.state.val_dataset is not None
        )

        model_provided = hasattr(self.state, "model") and self.state.model is not None

        log_dir_provided = hasattr(self.state, "log_dir") and self.state.log_dir is not None

        return dataset_provided and model_provided and log_dir_provided

    def is_done(self) -> bool:
        return False


@gin.configurable("TestModelSK")
class TestModelSK(PipelineStage):
    """Test a model using PyTorch Lightning API"""

    name = "Test with SK API"

    def __init__(
        self,
        state: PipelineState,
        accelerator: str = "cpu",
        num_workers: int = 1,
    ) -> None:
        super().__init__()
        self.state = state
        self.accelerator = accelerator
        self.done = False
        self.num_workers = num_workers
        self.columns = self.state.columns

    def run(self) -> Any:
        # Assumes we have executed `runnable` to check
        # either parent pipeline or arguments provide data
        assert self.state.dataset_class is not None
        test_dataset: ICUVariableLengthDataset = self.state.dataset_class(
            self.state.data_path, split="test"
        )

        if hasattr(self.state.train_dataset, "scaler"):
            test_dataset.set_scaler(self.state.train_dataset.scaler)  # type: ignore

        test_rep, test_label = test_dataset.get_data_and_labels(columns=self.columns)

        assert self.state.val_dataset is not None
        val_rep, val_label = self.state.val_dataset.get_data_and_labels(columns=self.columns)

        val_metrics = self.state.model_wrapper.evaluate(
            val_rep, val_label, "val", wandb_project=self.state.wandb_project
        )
        print_and_save_metrics(
            self.state.log_dir, val_metrics, prefix="val", print_metrics=True, strip_keys=0
        )

        test_metrics = self.state.model_wrapper.evaluate(
            test_rep, test_label, "test", wandb_project=self.state.wandb_project
        )
        print_and_save_metrics(
            self.state.log_dir, test_metrics, prefix="test", print_metrics=True, strip_keys=0
        )
        del test_dataset.h5_loader.lookup_table
        self.done = True

    def runnable(self) -> bool:
        # Check if we can load the dataset given parent information
        data_provided = (
            hasattr(self.state, "dataset_class")
            and self.state.dataset_class is not None
            and hasattr(self.state, "data_path")
            and self.state.data_path is not None
            and hasattr(self.state, "train_dataset")
        )

        model_provided = hasattr(self.state, "model") and self.state.model is not None

        log_provided = hasattr(self.state, "log_dir") and self.state.log_dir is not None

        return data_provided and model_provided and log_provided

    def is_done(self) -> bool:
        return self.done
