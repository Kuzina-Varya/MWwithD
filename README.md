# Продвинутые методы машинного обучения — лабораторные №1–5

В репозитории собраны полностью исполняемые решения лабораторных работ для
варианта 10. Каждая работа запускается из отдельного Jupyter Notebook: он
получает или создаёт данные, выполняет расчёты, сохраняет модели, изображения,
таблицы и формирует аналитический отчёт. Большие данные и модели отслеживаются
через DVC, а небольшие отчётные артефакты хранятся в Git.

## Лабораторная работа №1 — Iris и воспроизводимый Pipeline

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

Изменения следует фиксировать в отдельной ветке после проверки результатов:

```powershell
git status --short
git add <проверенные файлы лабораторной>
git commit -m "Complete lab 1 for Iris variant 10"
```

После push ветка объединяется с `main` через Pull Request. Notebook не создаёт
коммиты автоматически, поэтому перед фиксацией можно проверить изменения и не
добавить в историю посторонние файлы.

## Воспроизводимость

- Для случайных операций применяется `random_state=42`.
- Зависимости зафиксированы в `requirements.txt`.
- Импутация, масштабирование и модель объединены в Pipeline.
- GridSearchCV не использует тестовую выборку.
- Модель проверяется повторной загрузкой и сравнением предсказаний.
- Данные и модель контролируются DVC.

Фактические метрики последнего выполнения находятся в
`reports/model_metrics.json`, краткий вывод — в `reports/lab1_summary.md`.

## Лабораторная работа №2 — пропуски, выбросы и отбор признаков

Главный файл — `notebooks/lab2_iris_variant10.ipynb`. Для Iris воспроизводимо
создаются 15% пропусков в трёх числовых колонках по механизму **MCAR**: позиции
выбираются случайно и независимо от признаков и класса. Такой механизм нужен,
чтобы сравнение импутеров не смешивалось с дополнительной зависимостью MAR.
После стратифицированного split 80/20 четыре учебных выброса добавляются только
в train; test остаётся неизменным до финальной оценки.

### Что делает Notebook №2

1. Сравнивает `SimpleImputer`, `KNNImputer` и `IterativeImputer` на пяти фолдах.
2. Визуализирует восстановленные распределения признаков.
3. Сравнивает вариант без LOF и train-only `LOFTrainSampler`; test при поиске
   выбросов не используется.
4. Создаёт 14 признаков через `PolynomialFeatures`.
5. Сравнивает отсутствие отбора, RFE и L1-отбор.
6. Подбирает `C` логистической регрессии по CV, сохраняет Pipeline и один раз
   оценивает его на test.

По CV выбран KNNImputer (accuracy 0,9750), от LOF отказались, потому что он не
улучшил ROC-AUC, а лучшим отбором стал RFE с 8 признаками и CV ROC-AUC 0,9990.
На test получены accuracy 0,9667, F1-macro 0,9666 и ROC-AUC 0,9933: ошибочно
классифицирован один объект из 30.

### Где хранятся материалы №2

```text
data/raw/iris_lab2.csv(.dvc)             Iris с MCAR-пропусками
data/processed/lab2/train.csv(.dvc)      train с учебными выбросами
data/processed/lab2/test.csv(.dvc)       неизменённый отложенный test
models/iris_lab2_pipeline.pkl(.dvc)      полный переносимый Pipeline
reports/lab2/figures/                    пропуски, LOF, ROC и матрица ошибок
reports/lab2/imputation_comparison.csv   сравнение трёх импутеров
reports/lab2/lof_comparison.csv          сравнение с LOF и без него
reports/lab2/feature_selection_comparison.csv  сравнение RFE и L1
reports/lab2/selected_features.csv       выбранные полиномиальные признаки
reports/lab2/grid_search_results.csv     результаты GridSearchCV
reports/lab2/model_metrics.json          машиночитаемые итоговые метрики
reports/lab2/lab2_summary.md             аналитический отчёт
```

Для воспроизведения выполните все ячейки Notebook №2. Он самостоятельно
обновляет данные, модель, DVC-метаданные, таблицы, изображения и отчёт.

## Лабораторная работа №3 — бустинги, Optuna, SHAP и стекинг

Главный файл — `notebooks/lab3_iris_variant10.ipynb`. На одинаковой
пятифолдовой CV сравниваются XGBoost, LightGBM и CatBoost. У baseline-моделей
оставлены библиотечные гиперпараметры по умолчанию; явно заданы только seed,
число потоков и режим вывода. Это отделяет честное исходное сравнение от
последующей оптимизации.

### Что делает Notebook №3

1. Проверяет баланс трёх классов Iris и обосновывает отказ от SMOTE/весов.
2. Сравнивает три библиотеки по ROC-AUC, F1-macro и времени обучения.
3. Выполняет ровно 50 trials Optuna для лучшей библиотеки без доступа к test.
4. Строит SHAP summary, bar, два dependence plot и три waterfall для разных
   объектов, а не для одного примера.
5. Сравнивает базовый и оптимизированный CatBoost со `StackingClassifier` из
   LightGBM, RandomForest и SVC с `cv=5`.
6. Обучает выбранную по CV модель на полном train и один раз оценивает test.

Лучшим baseline стал CatBoost с CV ROC-AUC 0,9990. Лучший trial Optuna и
оптимизированный CatBoost достигли CV ROC-AUC 1,0000. На test итоговая модель
получила accuracy/F1-macro 0,9333 и ROC-AUC 0,9900; допущены две ошибки из 30.
SHAP показывает наибольшую роль `petal_length` и `petal_width` для virginica.

### Где хранятся материалы №3

```text
data/raw/iris_lab3.csv(.dvc)             исходный Iris
data/processed/lab3/train.csv(.dvc)      тренировочная часть
data/processed/lab3/test.csv(.dvc)       отложенный test
models/iris_lab3_model.pkl(.dvc)         оптимизированный CatBoost
reports/lab3/figures/                    EDA, Optuna, SHAP, ROC и ошибки
reports/lab3/boosting_comparison.csv     сравнение XGB/LGBM/CatBoost
reports/lab3/model_comparison.csv        baseline, optimized и stacking
reports/lab3/optuna_trials.csv           все 50 trials
reports/lab3/best_optuna_params.json     лучшие параметры
reports/lab3/shap_feature_importance.csv глобальная SHAP-важность
reports/lab3/model_metrics.json          итоговые метрики
reports/lab3/lab3_summary.md             аналитический отчёт
```

Полный повторный запуск Notebook №3 пересоздаёт все результаты. Optuna и SHAP
занимают больше времени, чем лабораторные №1–2.

## Лабораторная работа №4 — представления текста ISEAR

Главный файл — `notebooks/lab4_isear_variant10.ipynb`. Используется
`savalera/isear-from-original`, конфигурация `filtered`: 7 319 английских
описаний семи эмоций и исходный split 5 853/732/734. Сравниваются требуемые
три способа представления текста: TF-IDF, статический
`fasttext-wiki-news-subwords-300` и Sentence-Transformer
`all-MiniLM-L6-v2`. GloVe в итоговой реализации не используется.

### Что делает Notebook №4

1. Анализирует баланс классов и длины текстов.
2. Для TF-IDF и FastText приводит текст к нижнему регистру, удаляет пунктуацию
   и английские стоп-слова; контекстной модели передаёт минимально очищенный текст.
3. Подбирает `C` логистической регрессии отдельно для каждого представления.
4. Выбирает метод только по validation F1-macro и ROC-AUC.
5. Переобучает кандидатов на train+validation и использует test только для
   финального сравнения.
6. Для лучшего представления строит PCA → t-SNE и PCA → UMAP.

FastText даёт среднее покрытие словаря 0,973. Лучшим на validation стал
Sentence-Transformers: F1-macro 0,6244 и ROC-AUC 0,9014. На test он получил
accuracy 0,6008, F1-macro 0,6040 и ROC-AUC 0,8881. Частичное пересечение
эмоций на t-SNE/UMAP ожидаемо: близкие эмоциональные ситуации используют
похожую лексику и контекст.

### Где хранятся материалы №4

```text
data/raw/isear_lab4.csv(.dvc)                  очищенный ISEAR
data/processed/lab4/{train,validation,test}.csv(.dvc)  исходный split
models/isear_lab4_classifier.pkl(.dvc)         классификатор и метаданные
reports/lab4/figures/dataset_overview.png      классы и длины текстов
reports/lab4/figures/method_comparison.png     сравнение представлений
reports/lab4/figures/confusion_matrix.png      матрица ошибок
reports/lab4/figures/tsne_umap_embeddings.png  PCA → t-SNE/UMAP
reports/lab4/*_validation.csv                  подбор каждого метода
reports/lab4/method_comparison.csv             сравнение на validation
reports/lab4/test_comparison.csv               финальные test-метрики
reports/lab4/model_metrics.json                итоговая конфигурация
reports/lab4/lab4_summary.md                   аналитический отчёт
```

При первом запуске `gensim` загружает архив FastText размером около 1 ГБ, а
Sentence-Transformers — `all-MiniLM-L6-v2`; затем используются локальные кэши.
GPU выбирается автоматически, текущие результаты рассчитаны на CPU.

## Лабораторная работа №5 — PySpark, AutoML и MLflow

Главный файл — `notebooks/lab5_synthetic_regression_variant10.ipynb`. Notebook
создаёт синтетическую регрессию из 3 млн строк и 20 признаков, очищает данные в
PySpark, обучает два AutoGluon-запуска и регистрирует их в локальном MLflow.

### Что делает Notebook №5

1. Создаёт 2 999 000 уникальных наблюдений, 1 000 дубликатов и 6 000 пропусков.
2. В Spark удаляет дубликаты, выполняет медианную импутацию, приводит поля к
   `double` и добавляет сумму, среднее, L2-норму и взаимодействие признаков.
3. До pandas выполняет Spark split 80/20. Из train берёт 120 000 строк для
   AutoML, из test — 50 000 строк только для финальной оценки.
4. Запускает `medium_quality` и `deployment_optimized` с лимитом 120 секунд.
5. Сравнивает запуски только по внутреннему `score_val` AutoGluon. В
   leaderboard отсутствуют test-метрики; test не влияет на выбор модели.
6. После выбора один раз рассчитывает test RMSE/MAE/R², сравнивает результат с
   RandomForest baseline, анализирует остатки и permutation importance.
7. Сохраняет два запуска, параметры, validation-метрики, модели и версии в
   MLflow; скриншот UI фильтруется по тегу корректной схемы выбора.

По validation выбран `deployment_optimized` с `WeightedEnsemble_L2`: RMSE
20,8990 против 20,9222 у второго запуска. На отдельном test получены RMSE
20,8171, MAE 16,5505 и R² 0,9881. RandomForest baseline имеет RMSE 50,2179,
поэтому AutoML уменьшил ошибку на 58,55%. Среднее и медиана остатков близки к
нулю, то есть выраженного систематического смещения нет.

### Где хранятся материалы №5

```text
data/raw/lab5_synthetic_regression.parquet(.dvc)  3 млн исходных строк
data/processed/lab5/automl_train.parquet(.dvc)    120 000 train-строк
data/processed/lab5/automl_test.parquet(.dvc)     50 000 test-строк
models/lab5_autogluon/                            два predictor и baseline
models/lab5_autogluon.dvc                         DVC-метаданные моделей
reports/lab5/figures/data_overview.png            цель и объёмы данных
reports/lab5/figures/model_analysis.png           метрики, прогнозы, importance
reports/lab5/figures/mlflow_ui.png                два отфильтрованных запуска
reports/lab5/automl_run_comparison.csv            validation-сравнение запусков
reports/lab5/leaderboard_*.csv                    leaderboard без test
reports/lab5/test_metrics.csv                     финальный test и baseline
reports/lab5/feature_importance.csv               permutation importance
reports/lab5/residual_statistics.csv              статистика остатков
reports/lab5/spark_schema.csv                     итоговая схема Spark
reports/lab5/model_metrics.json                   машиночитаемые итоги
reports/lab5/lab5_summary.md                      аналитический отчёт
mlruns/                                           локальное хранилище, не в Git
```

На Windows Notebook автоматически применяет короткие ASCII-пути для Spark,
поэтому репозиторий может находиться в каталоге с кириллицей. Полные Spark
train/test не выгружаются в pandas, чтобы не создавать лишние копии данных и
не превышать доступную память.

## Как запустить любую лабораторную

После установки зависимостей откройте нужный файл в `notebooks/`, выберите
ядро проекта и выполните **Restart Kernel and Run All Cells**. Для запуска без
интерфейса замените имя файла в команде:

```powershell
jupyter nbconvert --to notebook --execute --inplace `
  notebooks\lab2_iris_variant10.ipynb `
  --ExecutePreprocessor.timeout=-1
```

После выполнения проверьте `git status --short`, `dvc status`, созданный
`reports/labN/labN_summary.md` и отсутствие ошибок в ячейках. В Git следует
добавлять только проверенные метаданные DVC, Notebook и отчётные артефакты;
локальные данные, модели, `.venv`, кэши и `mlruns` исключены правилами Git.
