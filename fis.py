import click

from commands import classify, detect
from pipelines import cleaning_pipeline
from shared.defintions import ResultDir


@click.group()
def cli() -> None:
    """A cli for running experiments published in the paper 'Tensor-to-Tensor Models with Fast Iterated Sum Features'.

    Authors: Joscha Diehl, Rasheed Ibraheem, Leonard Schmitz, and Yue Wu.
    """
    pass


@cli.group()
def clf() -> None:
    """
    Entry point for running classification pipelines.
    """
    pass


clf.add_command(classify.train)
clf.add_command(classify.hpsearch)
clf.add_command(classify.analyse)
clf.add_command(classify.random)
clf.add_command(classify.select)
clf.add_command(classify.compare)


@cli.group()
def ad() -> None:
    """
    Entry point for running anomaly detection pipelines.
    """
    pass


ad.add_command(detect.train)
ad.add_command(detect.analyse)
ad.add_command(detect.select)
ad.add_command(detect.backbone)
ad.add_command(detect.ldim)
ad.add_command(detect.topn)


@cli.command()
@click.option(
    "--exclude",
    "-e",
    multiple=True,
    type=click.Choice(ResultDir, case_sensitive=True),
    help="Exclude one or more dirs from the list of dirs set up for content deletion.",
)
def clean(exclude: str) -> None:
    """
    Run the clean up pipeline to delete all files in the result dirs.
    """

    directories = [d for d in ResultDir if d not in exclude]

    if click.confirm(
        f"Do you want to continue? All files in {', '.join(directories)} will be DELETED!",
        abort=True,
    ):
        click.echo("Deleting files...")
        cleaning_pipeline(directories=directories)

    return None
