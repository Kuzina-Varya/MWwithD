# MWwithD

Структура проекта для работы с данными, ноутбуками, исходным кодом, моделями и отчётами.

## Запуск pipeline

Главный файл запуска находится в корне репозитория:

```text
MWwithD/run_pipeline.py
```

Перед первым запуском активируйте виртуальное окружение и установите зависимости:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Полный запуск из корня репозитория:

```powershell
python run_pipeline.py
```

Pipeline последовательно:

1. загружает датасет Titanic с Kaggle;
2. разделяет данные на обучающую и тестовую выборки;
3. выполняет GridSearchCV и обучает лучшую модель;
4. оценивает модель на тестовой выборке и выводит метрики в терминал.

Если `data/raw/titanic.csv` уже загружен, повторную загрузку можно пропустить:

```powershell
python run_pipeline.py --skip-download
```

Доступные параметры:

```powershell
python run_pipeline.py --skip-download --test-size 0.2 --cv-folds 5 --random-state 42
```

Справка по параметрам:

```powershell
python run_pipeline.py --help
```

После выполнения создаются или обновляются:

- `data/processed/train.csv` — обучающая выборка;
- `data/processed/test.csv` — тестовая выборка;
- `models/titanic_pipeline.joblib` — обученный pipeline;
- `reports/model_metrics.json` — итоговые метрики и лучшие параметры;
- `reports/grid_search_results.csv` — результаты всех комбинаций GridSearchCV.
