"""Обучение и оценка классификационного pipeline для данных Titanic."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
import tempfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.transformers import MissingValueCountTransformer


DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "titanic_pipeline.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
DEFAULT_GRID_RESULTS_PATH = PROJECT_ROOT / "reports" / "grid_search_results.csv"

TARGET_COLUMN = "Survived"
NUMERIC_SOURCE_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
INPUT_FEATURES = NUMERIC_SOURCE_FEATURES + CATEGORICAL_FEATURES
NUMERIC_FEATURES = NUMERIC_SOURCE_FEATURES + ["MissingCount"]


def parse_args() -> argparse.Namespace:
    """Получить пути и параметры обучения из командной строки."""
    parser = argparse.ArgumentParser(
        description="Обучить логистическую регрессию для предсказания выживания."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--grid-results-output", type=Path, default=DEFAULT_GRID_RESULTS_PATH
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def validate_dataset(data: pd.DataFrame, dataset_name: str) -> None:
    """Проверить наличие признаков и корректность целевой переменной."""
    required_columns = set(INPUT_FEATURES + [TARGET_COLUMN])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"В выборке {dataset_name} отсутствуют столбцы: "
            f"{sorted(missing_columns)}"
        )
    if data.empty:
        raise ValueError(f"Выборка {dataset_name} пуста")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Целевая переменная {TARGET_COLUMN} в {dataset_name} содержит пропуски"
        )
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(
            f"Целевая переменная {TARGET_COLUMN} должна содержать только 0 и 1"
        )


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Создать pipeline предобработки и логистической регрессии."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("missing_count", MissingValueCountTransformer()),
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )


def build_grid_search(
    pipeline: Pipeline, cv_folds: int, random_state: int
) -> GridSearchCV:
    """Создать поиск гиперпараметров со стратифицированной кросс-валидацией."""
    if cv_folds < 2:
        raise ValueError("Количество folds должно быть не меньше двух")

    parameter_grid = {
        "preprocessor__numeric__imputer__strategy": ["median", "mean"],
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__class_weight": [None, "balanced"],
    }
    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cross_validation,
        refit=True,
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )


def calculate_metrics(
    y_true: pd.Series,
    predictions: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, object]:
    """Рассчитать метрики бинарной классификации на тестовой выборке."""
    matrix = confusion_matrix(y_true, predictions)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "test_rows": int(len(y_true)),
    }


def save_joblib_atomic(model: Pipeline, destination: Path) -> None:
    """Атомарно сохранить обученный pipeline в формате joblib."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".joblib", dir=destination.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        joblib.dump(model, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_json_atomic(payload: dict[str, object], destination: Path) -> None:
    """Атомарно сохранить словарь с метриками в JSON-файл."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        json.dump(payload, temporary_file, ensure_ascii=False, indent=2, sort_keys=True)
        temporary_file.write("\n")

    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_grid_results(search: GridSearchCV, destination: Path) -> None:
    """Сохранить ранжированные результаты всех комбинаций GridSearchCV."""
    raw_results = pd.DataFrame(search.cv_results_)
    results = pd.DataFrame(
        {
            "rank": raw_results["rank_test_score"].astype(int),
            "mean_roc_auc": raw_results["mean_test_score"],
            "std_roc_auc": raw_results["std_test_score"],
            "mean_fit_time": raw_results["mean_fit_time"],
            "parameters": [
                json.dumps(parameters, sort_keys=True)
                for parameters in raw_results["params"]
            ],
        }
    ).sort_values(["rank", "mean_fit_time"])

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
        results.to_csv(temporary_file, index=False, lineterminator="\n")

    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def train_and_evaluate(
    train_path: Path,
    test_path: Path,
    model_path: Path,
    metrics_path: Path,
    grid_results_path: Path,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """Подобрать pipeline на train, оценить на test и сохранить результаты."""
    if not train_path.is_file():
        raise FileNotFoundError(f"Обучающая выборка не найдена: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Тестовая выборка не найдена: {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    validate_dataset(train_data, "train")
    validate_dataset(test_data, "test")

    x_train = train_data[INPUT_FEATURES]
    y_train = train_data[TARGET_COLUMN]
    x_test = test_data[INPUT_FEATURES]
    y_test = test_data[TARGET_COLUMN]

    search = build_grid_search(
        pipeline=build_pipeline(random_state=random_state),
        cv_folds=cv_folds,
        random_state=random_state,
    )
    search.fit(x_train, y_train)
    best_pipeline = search.best_estimator_

    predictions = best_pipeline.predict(x_test)
    probabilities = best_pipeline.predict_proba(x_test)[:, 1]
    metrics = calculate_metrics(y_test, predictions, probabilities)
    metrics.update(
        {
            "model": "LogisticRegression",
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "generated_features": ["MissingCount"],
            "grid_search": {
                "best_cv_roc_auc": float(search.best_score_),
                "best_parameters": search.best_params_,
                "candidate_count": int(len(search.cv_results_["params"])),
                "cv_folds": cv_folds,
                "scoring": "roc_auc",
            },
            "random_state": random_state,
            "train_rows": int(len(train_data)),
        }
    )

    save_joblib_atomic(best_pipeline, model_path)
    save_json_atomic(metrics, metrics_path)
    save_grid_results(search, grid_results_path)
    return metrics


def main() -> None:
    """Запустить обучение модели с параметрами командной строки."""
    args = parse_args()
    metrics = train_and_evaluate(
        train_path=args.train.resolve(),
        test_path=args.test.resolve(),
        model_path=args.model_output.resolve(),
        metrics_path=args.metrics_output.resolve(),
        grid_results_path=args.grid_results_output.resolve(),
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    print(f"Модель сохранена: {args.model_output.resolve()}")
    print(f"Метрики сохранены: {args.metrics_output.resolve()}")
    print(f"Результаты GridSearchCV сохранены: {args.grid_results_output.resolve()}")
    print(f"Лучшие параметры: {metrics['grid_search']['best_parameters']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Матрица ошибок: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
