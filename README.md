# Diabetes Risk ML

Проект машинного обучения для прогнозирования риска развития сахарного диабета на основе набора данных Pima Indians Diabetes Database.

Проект основан на курсовой работе по дисциплине «Машинное обучение для решения задач здравоохранения».

## Возможности

Проект включает:

- подготовку и очистку данных;
- создание дополнительных признаков;
- обучение модели LightGBM;
- сохранение обученной модели;
- получение предсказания для нового пациента;
- автоматическое сохранение результата в JSON-файл.

## Структура проекта

```text
diabetes-risk-ml/
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   └── predict.py
├── data/
│   ├── diabetes.csv
│   ├── sample_patient.json
│   └── processed/
│       └── diabetes_processed.csv
├── models/
│   ├── lightgbm_diabetes.joblib
│   └── scaler.joblib
├── results/
├── configs/
├── scripts/
│   └── run.sh
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

## Требования

Для запуска проекта необходимы:

- Linux;
- Python 3.10–3.12;
- uv;
- системная библиотека OpenMP для LightGBM.

Для Ubuntu/Debian библиотека OpenMP устанавливается командами:

```bash
sudo apt update
sudo apt install -y libgomp1
```

## Установка зависимостей

Все Python-зависимости проекта описаны в файле `pyproject.toml`.

Для создания окружения и установки зависимостей необходимо выполнить:

```bash
uv sync
```

Точные версии пакетов фиксируются в файле `uv.lock`.

## Подготовка данных

Исходный датасет расположен по пути:

```text
data/diabetes.csv
```

Для подготовки данных необходимо выполнить:

```bash
uv run python src/prepare_data.py
```

Скрипт выполняет обработку нулевых значений, заполнение пропусков и создание дополнительных признаков.

Обработанный датасет сохраняется в файл:

```text
data/processed/diabetes_processed.csv
```

## Обучение модели

Для обучения модели LightGBM необходимо выполнить:

```bash
uv run python src/train.py
```

В ходе обучения данные разделяются на обучающую и тестовую выборки, признаки масштабируются с помощью `StandardScaler`, после чего обучается модель LightGBM.

После выполнения создаются файлы:

```text
models/lightgbm_diabetes.joblib
models/scaler.joblib
```

На тестовой выборке были получены следующие метрики:

- Accuracy: 0.8701
- ROC-AUC: 0.9480

## Получение предсказания

Пример входных данных расположен в файле:

```text
data/sample_patient.json
```

Для получения предсказания необходимо выполнить:

```bash
uv run python src/predict.py \
  --input data/sample_patient.json \
  --output results
```

Результат автоматически сохраняется в:

```text
results/prediction.json
```

Пример результата:

```json
{
    "prediction": 1,
    "risk": "Диабет",
    "probability": 0.5986
}
```

## Bash-скрипт

Для запуска основного сценария проекта используется Bash-скрипт:

```text
scripts/run.sh
```

Скрипт принимает два параметра:

- `--input` — путь к JSON-файлу с входными данными;
- `--output` — путь к каталогу для сохранения результата.

Пример запуска:

```bash
bash scripts/run.sh \
  --input data/sample_patient.json \
  --output results
```

Если каталог результатов отсутствует, он создаётся автоматически.

## Запуск проекта в Linux

Последовательность запуска проекта в Linux:

1. Установить системную зависимость LightGBM:

```bash
sudo apt update
sudo apt install -y libgomp1
```

2. Создать Python-окружение и установить зависимости:

```bash
uv sync
```

3. Подготовить данные:

```bash
uv run python src/prepare_data.py
```

4. Обучить модель:

```bash
uv run python src/train.py
```

5. Запустить основной сценарий:

```bash
bash scripts/run.sh \
  --input data/sample_patient.json \
  --output results
```

Проект был проверен в Linux-окружении WSL.

## Результаты

После запуска прогнозирования результат автоматически создаётся в каталоге `results`.

Файл результата:

```text
results/prediction.json
```

Проект предназначен для учебных целей и не является медицинским диагностическим инструментом.