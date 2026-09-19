"""Воспроизводимое стратифицированное разделение данных Titanic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "titanic.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COLUMN = "Survived"
ID_COLUMN = "PassengerId"
REQUIRED_COLUMNS = {
    ID_COLUMN,
    TARGET_COLUMN,
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}


def parse_args() -> argparse.Namespace:
    """Получить параметры запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description="Разделить данные Titanic на обучающую и тестовую выборки."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def validate_data(data: pd.DataFrame, test_size: float) -> None:
    """Проверить структуру данных и допустимость параметра test_size."""
    if not 0 < test_size < 1:
        raise ValueError("test_size должен находиться строго между 0 и 1")

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Отсутствуют обязательные столбцы: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Исходный датасет пуст")
    if data[ID_COLUMN].isna().any():
        raise ValueError(f"Столбец {ID_COLUMN} содержит пропущенные значения")
    if data[ID_COLUMN].duplicated().any():
        raise ValueError(f"Значения столбца {ID_COLUMN} должны быть уникальными")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Столбец {TARGET_COLUMN} содержит пропущенные значения")
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"Столбец {TARGET_COLUMN} должен содержать только 0 и 1")
    if data[TARGET_COLUMN].value_counts().min() < 2:
        raise ValueError("Для стратификации каждый класс должен содержать минимум две строки")


def write_csv_atomic(data: pd.DataFrame, destination: Path) -> None:
    """Атомарно записать таблицу в CSV без сохранения индекса."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".csv",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        data.to_csv(temporary_file, index=False, lineterminator="\n")

    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_atomic(payload: dict[str, object], destination: Path) -> None:
    """Атомарно записать словарь в JSON-файл."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        json.dump(payload, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")

    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def target_distribution(data: pd.DataFrame) -> dict[str, float]:
    """Рассчитать относительную долю каждого класса целевой переменной."""
    proportions = data[TARGET_COLUMN].value_counts(normalize=True).sort_index()
    return {str(class_label): float(value) for class_label, value in proportions.items()}


def split_data(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Path, Path, Path]:
    """Проверить и разделить данные, вернуть пути к созданным файлам."""
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Исходный датасет не найден: {input_path}")

    data = pd.read_csv(input_path)
    validate_data(data, test_size)

    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[TARGET_COLUMN],
        shuffle=True,
    )
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    overlap = set(train_data[ID_COLUMN]).intersection(test_data[ID_COLUMN])
    if overlap:
        raise RuntimeError(f"Выборки пересекаются по столбцу {ID_COLUMN}")
    if len(train_data) + len(test_data) != len(data):
        raise RuntimeError("При разделении датасета были потеряны строки")

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    metadata_path = output_dir / "split_metadata.json"
    write_csv_atomic(train_data, train_path)
    write_csv_atomic(test_data, test_path)

    try:
        source_name = input_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source_name = str(input_path)

    metadata: dict[str, object] = {
        "source": source_name,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "random_state": random_state,
        "test_size": test_size,
        "rows": {
            "source": len(data),
            "train": len(train_data),
            "test": len(test_data),
        },
        "target_distribution": {
            "source": target_distribution(data),
            "train": target_distribution(train_data),
            "test": target_distribution(test_data),
        },
    }
    write_json_atomic(metadata, metadata_path)
    return train_path, test_path, metadata_path


def main() -> None:
    """Запустить разделение данных с параметрами командной строки."""
    args = parse_args()
    train_path, test_path, metadata_path = split_data(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Обучающая выборка сохранена: {train_path}")
    print(f"Тестовая выборка сохранена: {test_path}")
    print(f"Метаданные разделения сохранены: {metadata_path}")


if __name__ == "__main__":
    main()
