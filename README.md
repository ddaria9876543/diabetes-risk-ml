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
- сохранение результатов прогнозирования в JSON-файл.

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