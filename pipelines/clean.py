from typing import Iterable

from shared.helpers import get_logger
from steps import clean_dir


def cleaning_pipeline(directories: Iterable[str]) -> None:
    logger = get_logger(__name__)

    logger.info("Cleaning pipeline has started.")

    logger.info(
        f"All files/directories in {', '.join([d for d in directories])} will be recursively DELETED!"
    )

    for d in directories:
        clean_dir(directory=d)

    logger.info("Cleaning pipeline finished successfully.")

    return None
