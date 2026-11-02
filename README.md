# Лабораторная работа №1 — Iris, вариант 10

Воспроизводимое решение первой лабораторной работы «Введение и
воспроизводимость исследований». Главный исполняемый документ — Jupyter
Notebook: он загружает данные, создаёт пропуски, проводит анализ, обучает
Pipeline, оценивает модель и раскладывает результаты по папкам проекта.

## Задание варианта

- Датасет: Iris из `sklearn.datasets`.
- Задача: многоклассовая классификация.
- Целевая переменная: `target` — вид ириса.
- Признаки: четыре числовых признака.
- Кастомный трансформер: отношение `petal_length / petal_width`.
- Так как Iris не содержит пропусков, Notebook воспроизводимо создаёт 10%
  пропусков в `sepal_length` и `petal_length`.

## Где что хранится

```text
MWwithD/
├── data/
│   ├── raw/
│   │   ├── iris_with_missing.csv       исходный Iris с пропусками (локально)
│   │   └── iris_with_missing.csv.dvc   DVC-метаданные исходных данных
│   └── processed/
│       ├── train.csv                    обучающая выборка (локально)
│       ├── train.csv.dvc                DVC-метаданные train
│       ├── test.csv                     тестовая выборка (локально)
│       └── test.csv.dvc                 DVC-метаданные test
├── notebooks/
│   └── lab1_iris_variant10.ipynb        главный исполняемый Notebook
├── models/
│   ├── iris_pipeline.pkl                переносимый полный Pipeline (локально)
│   └── iris_pipeline.pkl.dvc            DVC-метаданные модели
├── reports/
│   ├── figures/
│   │   ├── class_distribution.png       распределение классов
│   │   ├── missing_values.png           доли пропусков
│   │   ├── correlation_matrix.png       матрица корреляций
│   │   ├── feature_distributions.png    распределения признаков
│   │   ├── confusion_matrix.png         матрица ошибок
│   │   ├── roc_curves.png               ROC-кривые трёх классов
│   │   ├── project_structure.png         структура проекта для отчёта
│   │   ├── metrics_summary.png           сводка метрик для отчёта
│   │   └── dvc_status.png                результат dvc status для отчёта
│   ├── model_metrics.json               итоговые числовые метрики
│   ├── classification_report.csv        precision, recall и F1 по классам
│   ├── grid_search_results.csv           результаты GridSearchCV
│   └── lab1_summary.md                   автоматически созданный краткий отчёт
├── .dvc/                                 конфигурация DVC
├── .dvcignore                            исключения DVC
├── .gitignore                            исключения Git
├── requirements.txt                      зафиксированные зависимости
└── README.md                             описание проекта и инструкция
```

Локальные файлы данных и модели не добавляются непосредственно в Git: их
содержимое контролирует DVC, а Git хранит небольшие файлы `*.dvc`. Изображения
и таблицы отчёта сохраняются непосредственно в репозитории.

## Что делает Notebook

1. Определяет корень проекта и создаёт недостающие каталоги.
2. Выводит версии Python и библиотек.
3. Загружает Iris и переименовывает признаки.
4. Вносит 10% пропусков в две колонки с `random_state=42`.
5. Сохраняет данные и графики исследовательского анализа.
6. Делит данные на train/test 80/20 с сохранением пропорций классов.
7. Создаёт `PetalRatioTransformer` на основе `BaseEstimator` и
   `TransformerMixin`.
8. Создаёт `ColumnTransformer` с медианной импутацией и `StandardScaler`.
   Категориальной ветки нет, поскольку в Iris нет категориальных признаков.
9. Собирает Pipeline с логистической регрессией.
10. Подбирает параметры через `GridSearchCV` с 5-фолдовой стратификацией.
11. Считает accuracy, F1-macro и ROC-AUC macro OvR на отложенном тесте.
12. Сохраняет полный Pipeline через `cloudpickle` и проверяет его повторной
    загрузкой. Этот формат сохраняет также класс трансформера из Notebook.
13. Добавляет данные и модель под контроль DVC и выводит `dvc status`.

Все преобразования обучаются только на тренировочной части, что исключает
утечку информации из тестовой выборки.

## Установка

Из корня репозитория в PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name mw-with-d --display-name "Python (MWwithD)"
```

## Запуск

```powershell
jupyter lab
```

Откройте `notebooks/lab1_iris_variant10.ipynb`, выберите ядро
`Python (MWwithD)` и выполните **Kernel → Restart Kernel and Run All Cells**.

Проверка без интерфейса:

```powershell
jupyter nbconvert --to notebook --execute notebooks/lab1_iris_variant10.ipynb `
  --output lab1_iris_variant10.ipynb --output-dir notebooks `
  --ExecutePreprocessor.timeout=600
```

## DVC и Git

Notebook выполняет `dvc add` для исходных данных, train/test и модели, затем
показывает `dvc status`. DVC remote пока не настроен, поэтому для переноса
данных на другой компьютер позднее потребуется внешнее хранилище.

После проверки результатов:

```powershell
git add .
git commit -m "Complete lab 1 for Iris variant 10"
```

Коммит не создаётся автоматически, чтобы перед фиксацией можно было проверить
Notebook, метрики и изображения.

## Воспроизводимость

- Для случайных операций применяется `random_state=42`.
- Зависимости зафиксированы в `requirements.txt`.
- Импутация, масштабирование и модель объединены в Pipeline.
- GridSearchCV не использует тестовую выборку.
- Модель проверяется повторной загрузкой и сравнением предсказаний.
- Данные и модель контролируются DVC.

Фактические метрики последнего выполнения находятся в
`reports/model_metrics.json`, краткий вывод — в `reports/lab1_summary.md`.

## Лабораторная работа №2

Вторая лабораторная находится в отдельной ветке `lab-2-iris` и выполняется из
Notebook:

```text
notebooks/lab2_iris_variant10.ipynb
```

Она сравнивает SimpleImputer, KNNImputer и IterativeImputer, исследует LOF
только на тренировочных данных, создаёт полиномиальные признаки и сравнивает
RFE с L1-отбором. Выбор каждого компонента выполняется по пятифолдовой
кросс-валидации, а итоговая оценка — на отложенном test.

### Файлы лабораторной №2

```text
data/
├── raw/
│   ├── iris_lab2.csv                 данные с 15% пропусков (локально)
│   └── iris_lab2.csv.dvc             DVC-метаданные
└── processed/lab2/
    ├── train.csv                     train с учебными выбросами (локально)
    ├── train.csv.dvc                 DVC-метаданные train
    ├── test.csv                      неизменённый отложенный test (локально)
    └── test.csv.dvc                  DVC-метаданные test

models/
├── iris_lab2_pipeline.pkl            полный переносимый Pipeline (локально)
└── iris_lab2_pipeline.pkl.dvc        DVC-метаданные модели

reports/lab2/
├── figures/                          графики пропусков, LOF, ROC и ошибок
├── imputation_comparison.csv         сравнение трёх импутеров
├── lof_comparison.csv                сравнение с LOF и без LOF
├── feature_selection_comparison.csv  сравнение без отбора, RFE и L1
├── selected_features.csv             список полиномиальных признаков
├── grid_search_results.csv           результаты GridSearchCV
├── classification_report.csv         метрики по классам
├── model_metrics.json                итоговые метрики и конфигурация
└── lab2_summary.md                    полный аналитический отчёт
```

Для выполнения откройте `notebooks/lab2_iris_variant10.ipynb` и выберите
**Restart Kernel and Run All Cells**. Notebook самостоятельно обновит данные,
модель, сравнительные таблицы, изображения, DVC-файлы и аналитический отчёт.

## Лабораторная работа №3

Третья лабораторная находится в ветке `lab-3-iris`. Главный файл:

```text
notebooks/lab3_iris_variant10.ipynb
```

Notebook сравнивает XGBoost, LightGBM и CatBoost, проводит 50 trials Optuna,
строит SHAP-графики и сравнивает оптимизированный бустинг со стекингом из
LightGBM, RandomForest и SVC. Test используется только для финальной оценки.

### Файлы лабораторной №3

```text
data/
├── raw/
│   ├── iris_lab3.csv                  чистый Iris (локально)
│   └── iris_lab3.csv.dvc              DVC-метаданные
└── processed/lab3/
    ├── train.csv                      тренировочная выборка (локально)
    ├── train.csv.dvc                  DVC-метаданные train
    ├── test.csv                       отложенная выборка (локально)
    └── test.csv.dvc                   DVC-метаданные test

models/
├── iris_lab3_model.pkl                финальная модель (локально)
└── iris_lab3_model.pkl.dvc            DVC-метаданные модели

reports/lab3/
├── figures/                            EDA, Optuna, SHAP, ROC и ошибки
├── boosting_comparison.csv             сравнение трёх бустингов
├── model_comparison.csv                бустинг против стекинга
├── optuna_trials.csv                   результаты 50 trials
├── optuna_param_importance.csv         важность гиперпараметров
├── best_optuna_params.json             лучшие параметры
├── shap_feature_importance.csv         глобальная SHAP-важность
├── classification_report.csv           метрики по классам
├── model_metrics.json                  итоговые метрики
└── lab3_summary.md                      аналитический отчёт
```

Для повторного выполнения откройте Notebook и выберите **Restart Kernel and
Run All Cells**. Он самостоятельно пересоздаст данные, запустит сравнение,
Optuna, SHAP, стекинг и обновит модель, DVC-файлы и отчёт.
