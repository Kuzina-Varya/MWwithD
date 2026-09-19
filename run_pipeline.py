"""Последовательный запуск полного pipeline проекта Titanic."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Получить параметры запуска pipeline из командной строки."""
    parser = argparse.ArgumentParser(
        description=(
            "Загрузить Titanic, разделить данные и обучить лучшую модель."
        )
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Использовать существующий data/raw/titanic.csv без загрузки с Kaggle.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def run_stage(name: str, module: str, arguments: list[str]) -> None:
    """Запустить этап в отдельном процессе и остановиться при ошибке."""
    command = [sys.executable, "-m", module, *arguments]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    print(f"\n=== {name} ===", flush=True)
    started_at = perf_counter()
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    elapsed = perf_counter() - started_at
    print(f"Этап завершён за {elapsed:.1f} с", flush=True)


def main() -> None:
    """Выполнить этапы pipeline в корректном порядке."""
    args = parse_args()
    shared_arguments = ["--random-state", str(args.random_state)]

    if not args.skip_download:
        run_stage(
            name="1/3. Загрузка исходных данных",
            module="src.data.download_titanic",
            arguments=[],
        )
    else:
        print("\n=== 1/3. Загрузка пропущена ===", flush=True)

    run_stage(
        name="2/3. Разделение на train и test",
        module="src.data.split_data",
        arguments=["--test-size", str(args.test_size), *shared_arguments],
    )
    run_stage(
        name="3/3. Подбор гиперпараметров и обучение",
        module="src.models.train_model",
        arguments=["--cv-folds", str(args.cv_folds), *shared_arguments],
    )
    print("\nPipeline успешно выполнен.", flush=True)


if __name__ == "__main__":
    main()
