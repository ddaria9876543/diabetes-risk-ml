import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.train import create_features, replace_invalid_zeros

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models/lightgbm_diabetes.joblib"
SCALER_PATH = PROJECT_ROOT / "models/scaler.joblib"
IMPUTER_PATH = PROJECT_ROOT / "models/imputer.joblib"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def predict_patient(patient: dict) -> dict:
    """Получает прогноз для одного пациента."""
    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        IMPUTER_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Не найден файл модели или предобработки: {file_path}"
            )

    missing_columns = [column for column in FEATURE_COLUMNS if column not in patient]

    if missing_columns:
        raise ValueError(f"Отсутствуют признаки: {missing_columns}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    features = pd.DataFrame(
        [patient],
        columns=FEATURE_COLUMNS,
    )

    features = replace_invalid_zeros(features)

    features = pd.DataFrame(
        imputer.transform(features),
        columns=FEATURE_COLUMNS,
    )

    features = create_features(features)

    features_scaled = scaler.transform(features)

    prediction = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])

    return {
        "prediction": prediction,
        "risk": "Диабет" if prediction == 1 else "Нет диабета",
        "probability": round(probability, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diabetes risk prediction")

    parser.add_argument(
        "--input",
        required=True,
        help="Path to patient JSON file",
    )

    parser.add_argument(
        "--output",
        default="results",
        help="Directory for prediction results",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    with input_path.open(encoding="utf-8") as file:
        patient = json.load(file)

    result = predict_patient(patient)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "prediction.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("Результат прогнозирования:")
    print(json.dumps(result, ensure_ascii=False, indent=4))
    print(f"Результат сохранён: {output_path}")


if __name__ == "__main__":
    main()
