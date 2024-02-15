# Dynamic Survival Analysis for Early Event Prediction

Code used to generate the results in the paper "Dynamic Survival Analysis for Early Event Prediction".

## Data

We base our pipeline on the work of Yèche et al. in Temporal Label Smoothing. For instructions on
how to obtain access to the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [HiRID](https://physionet.org/content/hirid/1.1.1/) datasets, as well as how to preprocess them, please
consult the repository here: [https://github.com/ratschlab/tls](https://github.com/ratschlab/tls).

Follow instructions in `tls` repository to download and preprocess the datasets. The resulting `.h5` files in the `ml_stage` working directory of the preprocessing are the designated inputs to our pipeline.

## Setup

Setup the conda environment from `environment_linux.yml`, activate the environment, and then install the `dsaeep` package by running `pip install -e ./dsaeep` in the root directory of this repository.

Runs can be started with:
```bash
python -m dsaeep.scripts.train_sequence_model \
    -g {config_path} \
    -l {log_directory} \
    --seed {seed}
```

## Structure

- `dsaeep` contains the code for the dynamic survival analysis pipeline. Training models, evaluating models (time-step and event-level)
- `config` contains example gin configurations to run the pipeline together with the scripts in `dsaeep/dsaeep/scripts`