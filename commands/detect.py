import click

from pipelines import (
    ad_analysis_pipeline,
    ad_training_pipeline,
    backbone_robustness_pipeline,
    latent_dim_robustness_pipeline,
    topn_mean_robustness_pipeline,
)
from shared.defintions import (
    MVTECCategory,
    Semiring,
    TreeType,
)
from shared.params import (
    AD_BACKBONE_HYPERPARAMS,
    AD_LATENT_DIM_HYPERPARAMS,
    AD_TOPN_HYPERPARAMS,
    AD_TRAIN_HYPERPARAMS,
)

NUM_EPOCHS_OPT = dict(
    default=200,
    type=click.IntRange(min=1, max=None, min_open=False),
    show_default=True,
    help="""
    The number of epochs to train the model for.
    """,
)
ABLATION_OPT = dict(
    is_flag=True,
    help="Give this flag to run the ablation study.",
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


@click.command()
@click.option(
    "--category",
    default=MVTECCategory.CARPET,
    type=click.Choice(MVTECCategory, case_sensitive=True),
    show_default=True,
    help=f"""
    The category of mvtec data. Valid options are {", ".join(MVTECCategory)}.
    """,
)
@click.option("--semiring", **SEMIRING_OPT)
@click.option("--tree-type", **TREE_TYPE_OPT)
@click.option("--num-epochs", **NUM_EPOCHS_OPT)
@click.option("--ablation", **ABLATION_OPT)
def train(
    category: str, semiring: str, tree_type: str, num_epochs: int, ablation: bool
) -> None:
    "Run the training pipeline with an MVTEC data category."

    ad_training_pipeline(
        category=category,
        semiring=semiring,
        tree_type=tree_type,
        hyperparams=AD_TRAIN_HYPERPARAMS,
        num_epochs=num_epochs,
        ablation=ablation,
    )

    return None


@click.command()
@click.option("--ablation", **ABLATION_OPT)
def analyse(ablation: bool) -> None:
    """
    Run the analysis pipeline. This pipeline requires you have run
    the 'select' command.
    """
    ad_analysis_pipeline(ablation=ablation)
    return None


@click.command()
@click.option("--semiring", **SEMIRING_OPT)
@click.option("--tree-type", **TREE_TYPE_OPT)
@click.option("--num-epochs", **NUM_EPOCHS_OPT)
@click.option(
    "--exclude",
    multiple=True,
    type=click.Choice(MVTECCategory, case_sensitive=True),
    help="Exclude one or more mvtec categories from anomaly detection training.",
)
@click.option("--ablation", **ABLATION_OPT)
def select(
    semiring: str, tree_type: str, num_epochs: int, exclude: str, ablation: bool
) -> None:
    """
    Run some selected experiments that are required to run the
    'analyse' command. Running this command run the anomaly detection
    training for all the mvtec data categories except for those
    specified with --exclude or -e option.
    """

    categories = [c for c in MVTECCategory if c not in exclude]

    if categories:
        for c in categories:
            ad_training_pipeline(
                category=c,
                semiring=semiring,
                tree_type=tree_type,
                hyperparams=AD_TRAIN_HYPERPARAMS,
                num_epochs=num_epochs,
                ablation=ablation,
            )
    else:
        click.echo("all categories are excluded!")
        exit(1)

    return None


@click.command()
@click.option("--semiring", **SEMIRING_OPT)
@click.option("--tree-type", **TREE_TYPE_OPT)
@click.option("--num-epochs", **NUM_EPOCHS_OPT)
def backbone(semiring: str, tree_type: str, num_epochs: int) -> None:
    "Run the robustness of anomaly detection to resnet backbone pipeline."

    backbone_robustness_pipeline(
        semiring=semiring,
        tree_type=tree_type,
        hyperparams=AD_BACKBONE_HYPERPARAMS,
        num_epochs=num_epochs,
    )

    return None


@click.command()
@click.option("--semiring", **SEMIRING_OPT)
@click.option("--tree-type", **TREE_TYPE_OPT)
@click.option("--num-epochs", **NUM_EPOCHS_OPT)
def ldim(semiring: str, tree_type: str, num_epochs: int) -> None:
    "Run the robustness of anomaly detection to latent dimension pipeline."

    latent_dim_robustness_pipeline(
        semiring=semiring,
        tree_type=tree_type,
        hyperparams=AD_LATENT_DIM_HYPERPARAMS,
        num_epochs=num_epochs,
    )

    return None


@click.command()
@click.option("--semiring", **SEMIRING_OPT)
@click.option("--tree-type", **TREE_TYPE_OPT)
@click.option("--num-epochs", **NUM_EPOCHS_OPT)
def topn(semiring: str, tree_type: str, num_epochs: int) -> None:
    "Run the robustness of anomaly detection to using different mean of top n reconstruction errors."

    topn_mean_robustness_pipeline(
        semiring=semiring,
        tree_type=tree_type,
        hyperparams=AD_TOPN_HYPERPARAMS,
        num_epochs=num_epochs,
    )

    return None
