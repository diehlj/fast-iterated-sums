import click

from pipelines import (
    clf_analysis_pipeline,
    comparison_pipeline,
    hpsearch_pipeline,
    random_robustness_pipeline,
    training_pipeline,
)
from shared.defintions import (
    DataName,
    ModelMode,
    ResnetMode,
    SelectMode,
    Semiring,
    TreeType,
)
from shared.params import CLF_RANDOM_HYPERPARAMS, CLF_TRAIN_HYPERPARAMS, SWEEP_CONFIG

NUM_EPOCHS_OPT = dict(
    default=200,
    type=click.IntRange(min=1, max=None, min_open=False),
    show_default=True,
    help="""
    The number of epochs to train the model for.
    """,
)
DATA_NAME_OPT = dict(
    type=click.Choice(DataName, case_sensitive=True),
    default=DataName.CIFAR10,
    show_default=True,
    help=f"""Specify which data to use for training model.
    Input must be one of {", ".join(DataName)}.
        """,
)
MODEL_MODE_OPT = dict(
    type=click.Choice(ModelMode, case_sensitive=True),
    default=ModelMode.L23,
    show_default=True,
    help=f"""Specify which model or network architecture to use.
    Input must be one of {", ".join(ModelMode)}.
        """,
)
RESNET_MODE_OPT = dict(
    type=click.Choice(ResnetMode, case_sensitive=True),
    default=ResnetMode.RESNET20,
    show_default=True,
    help=f"""Specify which base model or network architecture to use.
    Input must be one of {", ".join(ResnetMode)}.
        """,
)
SEMIRING_OPT = dict(
    type=click.Choice(Semiring, case_sensitive=True),
    default=Semiring.MAXPLUS,
    show_default=True,
    help=f"""Specify which semiring to use to build tree.
    Input must be one of {", ".join(Semiring)}.
        """,
)
TREE_TYPE_OPT = dict(
    type=click.Choice(TreeType, case_sensitive=True),
    default=TreeType.RANDOM,
    show_default=True,
    help=f"""Specify which tree type to use for building model.
    Input must be one of {", ".join(TreeType)}.
        """,
)
WANDB_OPT = dict(
    is_flag=True,
    help="""Give this flag if you want to log metrics to wandb server when running
    the training pipeline. Note that this flag requires wandb.
        """,
)


@click.command()
@click.option("--data-name", **DATA_NAME_OPT)
@click.option("--model-mode", **MODEL_MODE_OPT)
@click.option("--resnet-mode", **RESNET_MODE_OPT)
@click.option("--semiring", **SEMIRING_OPT)
@click.option("--tree-type", **TREE_TYPE_OPT)
@click.option("--num-epochs", **NUM_EPOCHS_OPT)
@click.option("--include-wandb", **WANDB_OPT)
def train(
    data_name: str,
    resnet_mode: str,
    model_mode: str,
    semiring: str,
    tree_type: str,
    num_epochs: int,
    include_wandb: bool,
) -> None:
    "Run the training pipeline."

    training_pipeline(
        data_name=data_name,
        model_mode=model_mode,
        resnet_mode=resnet_mode,
        semiring=semiring,
        tree_type=tree_type,
        hyperparams=CLF_TRAIN_HYPERPARAMS,
        num_epochs=num_epochs,
        include_wandb=include_wandb,
    )

    return None


@click.command()
@click.option("--data-name", **DATA_NAME_OPT)
@click.option("--model-mode", **MODEL_MODE_OPT)
@click.option("--resnet-mode", **RESNET_MODE_OPT)
def hpsearch(
    data_name: str,
    resnet_mode: str,
    model_mode: str,
) -> None:
    "Run the hyperparameter pipeline. Note that this pipeline requires wandb."

    hpsearch_pipeline(
        data_name=data_name,
        model_mode=model_mode,
        resnet_mode=resnet_mode,
        hyperparams=SWEEP_CONFIG,
    )

    return None


@click.command()
@click.option("--data-name", **DATA_NAME_OPT)
def compare(data_name: str) -> None:
    """
    Run the comparison pipeline. This pipeline requires that you have run the
    'select' command in mode 'all_train'. It compares our developed methods
    to the past literature.
    """
    comparison_pipeline(data_name=data_name)

    return None


@click.command()
@click.option("--data-name", **DATA_NAME_OPT)
@click.option("--model-mode", **MODEL_MODE_OPT)
@click.option("--resnet-mode", **RESNET_MODE_OPT)
@click.option("--semiring", **SEMIRING_OPT)
@click.option("--num-epochs", **NUM_EPOCHS_OPT)
def random(
    data_name: str,
    model_mode: str,
    resnet_mode: str,
    semiring: str,
    num_epochs: int,
) -> None:
    """
    Run the tree randomness effect pipeline. This pipeline
    investigates the effect of tree randomness on the Fast Iterated
    Sums Layer.
    """
    random_robustness_pipeline(
        data_name=data_name,
        resnet_mode=resnet_mode,
        model_mode=model_mode,
        semiring=semiring,
        hyperparams=CLF_RANDOM_HYPERPARAMS,
        num_epochs=num_epochs,
    )

    return None


@click.command()
@click.option("--data-name", **DATA_NAME_OPT)
@click.option("--model-mode", **MODEL_MODE_OPT)
def analyse(
    data_name: str,
    model_mode: str,
) -> None:
    """
    Run the analysis pipeline. This pipeline requires you have run the
    'select' command in 'all_random' mode.
    """
    clf_analysis_pipeline(data_name=data_name, model_mode=model_mode)

    return None


@click.command()
@click.option(
    "--select-mode",
    type=click.Choice(SelectMode, case_sensitive=True),
    default=SelectMode.ALL_TRAIN,
    show_default=True,
    help=f"""Specify which selected experiments to run.
    Input must be one of {", ".join(SelectMode)}.
        """,
)
@click.option("--data-name", **DATA_NAME_OPT)
@click.option("--semiring", **SEMIRING_OPT)
@click.option("--tree-type", **TREE_TYPE_OPT)
@click.option("--num-epochs", **NUM_EPOCHS_OPT)
def select(
    select_mode: str,
    data_name: str,
    semiring: str,
    tree_type: str,
    num_epochs: int,
) -> None:
    """
    Run some selected pipelines that are required to run the
    'analyse' and 'compare' commands.
    """

    match select_mode:
        case SelectMode.ALL_TRAIN:
            for rm in ResnetMode:
                for mm in ModelMode:
                    if mm == ModelMode.BASE:
                        continue

                    click.echo(
                        f"Executing training pipeline with {data_name}, {rm} and {mm}"
                    )
                    training_pipeline(
                        data_name=data_name,
                        model_mode=mm,
                        resnet_mode=rm,
                        semiring=semiring,
                        tree_type=tree_type,
                        hyperparams=CLF_TRAIN_HYPERPARAMS,
                        num_epochs=num_epochs,
                        include_wandb=False,
                    )

        case SelectMode.ALL_RANDOM:
            for rm in ResnetMode:
                click.echo(
                    f"Executing the tree randomness robustness pipeline with {data_name} in ca_fisblock model mode"
                )
                random_robustness_pipeline(
                    data_name=data_name,
                    model_mode="ca_fisblock",
                    resnet_mode=rm,
                    semiring=semiring,
                    hyperparams=CLF_RANDOM_HYPERPARAMS,
                    num_epochs=num_epochs,
                )

        case _:
            raise ValueError(
                f"Invalid select_mode. Valid values are {', '.join(SelectMode)}"
            )

    return None
