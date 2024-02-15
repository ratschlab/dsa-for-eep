#!/usr/bin/env python3

# ===========================================================
#
# Run HiRID Preprocessing
# Partially from: https://github.com/ratschlab/HIRID-ICU-Benchmark/blob/master/icu_benchmarks/run.py
#
# ===========================================================
import argparse
import logging
import socket
from pathlib import Path

import coloredlogs
import torch

from dsaeep.pipeline import GenericPipeline
from dsaeep.train.pipeline import DLTrainPipeline, SetSeeds, SetupGin

# ========================
# GLOBAL
# ========================
LOGGING_LEVEL = logging.INFO

# ========================
# Argparse
# ========================
def parse_arguments(argv=None) -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    # ------------------
    # Preprocessing Options
    # ------------------
    parser.add_argument(
        "--seed",
        dest="seed",
        default=42,
        required=False,
        type=int,
        help="Seed for the train/val/test split",
    )
    parser.add_argument(
        "-g", "--gin_config_path", required=True, type=Path, help="Path to Gin configuration file"
    )
    parser.add_argument(
        "-l", "--log_dir", required=True, type=Path, help="Path to log directory for this run"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow log directory overwriting of results"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB Project; default is `None`, thus WandB is not used",
    )

    args = parser.parse_args(argv)
    return args


# ========================
# MAIN
# ========================
def main(argv=None):
    """Training Script procedure"""

    # get GPU availability
    cuda_available = torch.cuda.is_available()
    device_string = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    logging.info(40 * "=")
    logging.info("Start Training script")
    logging.info(f"Host: {socket.gethostname()}")
    logging.info(f"Torch device: {device_string}")
    if cuda_available:
        gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        logging.info(f"GPU Memory: {gpu_memory} GB")
    logging.info(40 * "=")

    # Parse CMD arguments
    args = parse_arguments(argv)

    # ------------------------
    # Pipeline
    # ------------------------
    arguments = vars(args)
    log_dir = arguments["log_dir"] / f"seed_{arguments['seed']}"

    seed_stage = SetSeeds(random_state=arguments["seed"])
    setup_gin_stage = SetupGin(gin_config_files=[arguments["gin_config_path"]])
    setup_pipeline = GenericPipeline([seed_stage, setup_gin_stage])
    setup_pipeline.run()

    train_pipeline = DLTrainPipeline(
        gin_config_files=[arguments["gin_config_path"]],
        log_dir=log_dir,
        overwrite=arguments["overwrite"],
        random_state=arguments["seed"],
        wandb_project=arguments["wandb_project"],
    )
    train_pipeline.run()

    # ------------------------
    # Cleanup
    # ------------------------
    logging.info(40 * "=")
    logging.info("Finished")
    logging.info(40 * "=")


# ========================
# SCRIPT ENTRY
# ========================
if __name__ == "__main__":

    # set logging
    logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s - %(levelname)s | %(message)s")
    coloredlogs.install(level=LOGGING_LEVEL)

    # run script
    main()
