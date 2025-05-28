# Tensor-to-Tensor Models with Fast Iterated Sum Features

This repository implements fast algorithms for computing certain iterated sums on image data, based on the mathematical framework of corner trees [Even-Zohar, Leng '21]. The implementation provides efficient PyTorch operations for computing these features, which can be used in deep learning models for image processing tasks.

## Overview
The core concept is based on generalizing iterated-sum signatures from 1D time-series data to 2D image data. This implementation focuses on a computationally efficient subset of these features that can be calculated in linear time using corner trees. In addition, we show how fast iterated sums features can be used in classification and anomaly detection tasks.


## Set up
1. Clone the repository by running
    ```
    git clone https://github.com/diehlj/fast-iterated-sums.git
    ```
1. Navigate to the root folder, create a python virtual environment by running
    ```
    python -m venv .venv
    ```
    > Note that Python 3.12.3 was used in this research. If you are using uv to manage your project, run
    >
    > ```uv venv .venv --python 3.12.3```



1. Activate the virtual environment by running
    ```
    source .venv/bin/activate
    ```
1. Prepare all modules and required directories by running the following:
    ```
    make setup
    ```

    > If you are using uv to manage your environment, run
    >
    > ```mkdir plots data history models hpruns```
    >
    > and then
    >
    > ```uv sync```

1. If you would like to track experiments and running hyperparameter sweeps, register on Weight and Biases platform, `wandb`, to obtain the following: `WANDB_API_KEY`, `WANDB_PROJECT_NAME`, and `WANDB_ENTITY_NAME`. Create a `.env` file in the project root directory and store these informatuion there. These will be used for experiment tracking.

## Usage
For convenience, we built a cli tool for running various experiments in our paper.

We call this tool `fis`.

There are three main commands in fis: `clf`, `ad`, and `clean`. You can check this by running:

```
fis --help
```
in your terminal.

We explain each of these commands as follows.

1. `clf` command provides entry to run all experiments that have to do with classification task. It also has subcommands for running specific experiment. You can check all the commands under `clf` by running the followin in your terminal:

    ```
    fis clf --help
    ```

    Running a subcommand under `clf` is simple. For instance, if you want to run the training pipeline with default arguments:

    ```
    fis clf train
    ```

    Note that you can see all arguments and options available to a subcommand by running:

    ```
    fis clf [SUBCOMMAND] --help
    ```

1. `ad` command provides entry to running all experiments corresponding to anomaly detection task on the texture images of the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). Just like `clf` command, `ad` has subcommands. You can see these by running:

    ```
    fis ad --help
    ```

    Running a subcommand of `ad` is easy. For example you can run the training pipeline using:

    ```
    fis ad train
    ```

    Note that all arguments/options under a subcommand can be checked by running:

    ```
    fis ad [SUBCOMMAND] --help
    ```

1. `clean` command **DELETES** all the contents of the following directories: `plots`, `history`, `models` and `wandb`. It cleans up the dev directories. To run it, use

    ```
    fis clean
    ```

    You can exclude one or more dir from content deletion, e.g., if you want to exlude the contents of plots from deletion, run:

    ```
    fis clean --exclude ./plots
    ```

    and more than one dirs:

    ```
    fis clean --exclude ./plots --exclude ./wandb
    ```


## Citation
Comming soon.
