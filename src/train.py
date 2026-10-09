from pathlib import Path

import joblib
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/diabetes_processed.csv")
MODEL_PATH = Path("models/lightgbm_diabetes.joblib")
SCALER_PATH = Path("models/scaler.joblib")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Обработанный датасет не найден: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = lgb.LGBMClassifier(
        random_state=42,
        verbose=-1,
    )

    print("Обучение модели LightGBM...")

    model.fit(
        X_train_scaled,
        y_train,
    )

    y_pred = model.predict(
        X_test_scaled
    )

    y_proba = model.predict_proba(
        X_test_scaled
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Нет диабета",
                "Диабет",
            ],
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    print()
    print(
        f"Модель сохранена: {MODEL_PATH}"
    )

    print(
        f"Scaler сохранён: {SCALER_PATH}"
    )


if __name__ == "__main__":
    main()