"""Загрузка датасета Titanic с Kaggle в каталог исходных данных проекта."""

from pathlib import Path
import shutil
import tempfile

import kagglehub


DATASET_HANDLE = "yasserh/titanic-dataset"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "titanic.csv"


def download_titanic() -> Path:
    """Загрузить последнюю версию датасета и вернуть путь к CSV-файлу."""
    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_files = sorted(dataset_dir.glob("*.csv"))

    if len(csv_files) != 1:
        names = ", ".join(path.name for path in csv_files) or "файлы отсутствуют"
        raise RuntimeError(
            f"Ожидался один CSV-файл в {dataset_dir}, "
            f"найдено {len(csv_files)}: {names}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=OUTPUT_PATH.parent, suffix=".csv", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        shutil.copyfile(csv_files[0], temporary_path)
        temporary_path.replace(OUTPUT_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)

    return OUTPUT_PATH


if __name__ == "__main__":
    path = download_titanic()
    print(f"Датасет Titanic сохранён: {path}")
