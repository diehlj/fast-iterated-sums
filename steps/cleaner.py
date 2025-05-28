import shutil
from pathlib import Path


def clean_dir(directory: str) -> None:
    plots_dir = Path(directory)

    try:
        plots_dir.mkdir(exist_ok=True)

        if not any(plots_dir.iterdir()):
            print(f"Directory '{directory}' is already empty")
            return

        for item in plots_dir.iterdir():
            if item.is_file():
                item.unlink()
                print(f"Removed file: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"Removed directory: {item}")

        print(f"Successfully cleaned directory: {directory}")

    except PermissionError:
        print(f"Permission denied: Unable to access {directory}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return None
