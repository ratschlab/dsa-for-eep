# ========================================
#
# Training Utilities
#
# ========================================
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import gin
import torch
import wandb
from coolname import generate_slug

# ========================================
#
# Gin
#
# ========================================
# Torch
gin.config.external_configurable(
    torch.nn.functional.binary_cross_entropy_with_logits, module="torch.nn.functional"
)


# ========================================
#
# WandB
#
# ========================================
def init_wandb(wandb_project: str, config: dict = None):
    """
    Initializes a wandb run if not already initialized.
    Generates a random run name using `coolname` package

    Parameter
    ---------
    wandb_project: str
        the name of the wandb project
    config: dict
        the configuration to log to wandb
    """
    if wandb_project is not None and wandb.run is None:
        logging.info(f"[init_wandb] Initializing wandb run")
        run_name = generate_slug(3)
        wandb.init(project=wandb_project, name=run_name, config=config)


# ========================================
#
# I/O
#
# ========================================
def gin_config_to_readable_dictionary(gin_config: dict) -> dict:
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B
    Ref: https://github.com/google/gin-config/issues/154

    Parameter:
    ----------
    gin_config:
        the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG

    Return
    ------
    dict
        the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():

        if key[1] == "gin.macro":
            name = key[0]
        else:
            # ignore package name
            # name = ".".join(key[1].split(".")[1:])
            name = f"{key[0] + '.' if key[0] != '' else ''}{key[1]}"

        values = gin_config[key]
        for k, v in values.items():

            if key[1] == "gin.macro":
                data[name] = v  # assumes macro has only single item assignment
            else:
                data[".".join([name, k])] = v

    return data


def save_gin_config_file(log_dir: Path, filename: str = "train_config.gin"):
    """
    Save the currently active GIN configuration

    Parameter
    ---------
    log_dir: Path
        directory to store config to
    """
    with open(os.path.join(log_dir, filename), "w") as f:
        f.write(gin.operative_config_str())


def print_and_save_metrics(
    log_dir: Path,
    metrics: dict[str, float],
    prefix: str = "test",
    print_metrics: bool = False,
    strip_keys: int = 0,
):
    """
    Save and print a dictionary of metrics

    Parameter
    ---------
    log_dir: Path
        path to save metrics file to
    metrics: dict[str, float]
        the results to store
    prefix: str
        the prefix for the final file
        default: "test" yields "test_metrics.pkl"
    print_metrics: bool
        use logging to print dict, default: False
    strip_keys: int
        strip first `strip_keys` number of characters
        from keys
    """
    if strip_keys > 0:
        metrics = {k[strip_keys:]: v for k, v in metrics.items()}

    filename = f"{prefix}_metrics.pkl"
    filepath = log_dir / filename
    with open(filepath, "wb") as f:
        pickle.dump(metrics, f)

    if print_metrics:
        logging.info(f"[RESULT] {prefix} Metrics")
        for k, v in metrics.items():
            logging.info(f"\t{k: <15}: {v:.4f}")


def load_pickle(path) -> Any:
    """
    Loads a pickled python object
    """
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return data


# ========================================
#
# Tensor Transformations
#
# ========================================
@torch.no_grad()
def softmax_binary_output_transform(
    output: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies softmax and collapses the last dimension

    Parameter
    ---------
    output: tuple[torch.Tensor, torch.Tensor]
        a tuple of predictions and labels
    """
    y_pred, y = output
    y_pred = torch.softmax(y_pred, dim=1)
    return y_pred[:, -1], y


@torch.no_grad()
def sigmoid_binary_output_transform(
    output: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies sigmoid on predictions

    Parameter
    ---------
    output: tuple[torch.Tensor, torch.Tensor]
        a tuple of predictions and labels
    """
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    return y_pred, y.long()


@torch.no_grad()
def softmax_multi_output_transform(
    output: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies softmax to the predictions

    Parameter
    ---------
    output: tuple[torch.Tensor, torch.Tensor]
        a tuple of predictions and labels
    """
    y_pred, y = output
    y_pred = torch.softmax(y_pred, dim=1)
    return y_pred, y


def get_failure_from_hazard(hazard_pred):
    """Converts hazard to cumulative failure predictions."""
    log_hs = torch.log(torch.sigmoid(hazard_pred))
    failure_curve = 1 - torch.exp(torch.cumsum(log_hs - hazard_pred, dim=-1))
    return failure_curve


@torch.no_grad()
def eep_output_transform(output, pred_horizon):
    """Transfrom for hazard prediction to EEP ones"""
    y_pred, y = output
    failure_preds = get_failure_from_hazard(y_pred)[..., pred_horizon]
    labeled = torch.any((y != -1), dim=-1)  # Whether there is a label
    labels = torch.any((y[:, :, :pred_horizon] == 1), dim=-1).to(
        torch.int32
    )  # Whether the label is positive
    labels[~labeled] = -1
    return failure_preds, labels


def identity_logit_transform(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return logits, labels


def binary_task_masked_select(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a mask selection of a sequence model output i.e.
    selects only unmasked time-points for loss computation

    Parameter
    ---------
    logits: torch.Tensor[batch, time, class]
        model output logits batch
    labels: torch.Tensor[batch, time]
        target labels
    mask: torch.Tensor[batch, time]
        mask
    """
    logits_selected = torch.masked_select(logits, mask.unsqueeze(-1))
    logits_flat = logits_selected.flatten()
    labels_flat = torch.masked_select(labels, mask)

    return logits_flat, labels_flat


def surv_task_mask_select(
    preds: torch.Tensor, labels_eep: torch.Tensor, labels_surv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a mask selection of a survival model output from EEP tasks.

    Parameter
    ---------
    logits: torch.Tensor[batch, time, class]
        model output logits batch
    labels_eep: torch.Tensor[batch, time]
        target labels for eep
    labels_surv: torch.Tensor[batch, time]
        target labels for survival likelihood
    """
    labeled = torch.any((labels_surv != -1), dim=-1)
    return torch.masked_select(preds, labeled), torch.masked_select(labels_eep, labeled)
