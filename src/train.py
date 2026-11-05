import hashlib
import logging
import subprocess
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data/diabetes.csv"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MODELS_DIR = PROJECT_ROOT / "models"

ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Конфигурация не найдена: {CONFIG_PATH}")

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Некорректный формат конфигурации")

    if "model" not in config or "training" not in config:
        raise ValueError("Отсутствуют разделы model или training")

    if config["model"].get("name") != "lightgbm":
        raise ValueError("Поддерживается только LightGBM")

    return config


def get_data_version() -> str:
    hash_value = hashlib.md5(
        DATA_PATH.read_bytes(),
        usedforsecurity=False,
    ).hexdigest()

    return hash_value


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.warning("Не удалось определить Git-коммит")
        return "unknown"

    return result.stdout.strip()


def replace_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for column in ZERO_COLUMNS:
        zero_count = int((result[column] == 0).sum())

        if zero_count > 0:
            logger.warning(
                "Признак %s: найдено %s нулевых значений",
                column,
                zero_count,
            )

        result[column] = result[column].replace(0, np.nan)

    return result


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["Insulin_BMI"] = result["Insulin"] * result["BMI"]
    result["Glucose_BMI"] = result["Glucose"] * result["BMI"]
    result["Glucose_Age"] = result["Glucose"] * result["Age"]
    result["BMI_Age"] = result["BMI"] * result["Age"]

    result["Insulin_Glucose"] = result["Insulin"] * result["Glucose"]

    result["Pregnancies_Age"] = result["Pregnancies"] * result["Age"]

    return result


def main() -> None:
    logger.info("Запуск обучения модели")

    config = load_config()
    model_config = config["model"]
    training_config = config["training"]

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Датасет не найден: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    logger.info("Датасет загружен: %s строк", len(df))

    if df.isnull().any().any():
        logger.warning("В исходном датасете обнаружены пропуски")

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_config["test_size"],
        random_state=training_config["random_state"],
        stratify=y,
    )

    logger.info(
        "Обучающая выборка: %s, тестовая выборка: %s",
        len(X_train),
        len(X_test),
    )

    X_train = replace_invalid_zeros(X_train)
    X_test = replace_invalid_zeros(X_test)

    # Медианы вычисляются только на обучающей выборке.
    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index,
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X.columns,
        index=X_test.index,
    )

    X_train = create_features(X_train)
    X_test = create_features(X_test)

    # Масштабирование также обучается только на train.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = lgb.LGBMClassifier(
        n_estimators=model_config["n_estimators"],
        learning_rate=model_config["learning_rate"],
        num_leaves=model_config["num_leaves"],
        random_state=model_config["random_state"],
        verbose=-1,
    )

    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("diabetes-risk-experiments")

    run_name = (
        f"trees_{model_config['n_estimators']}_"
        f"lr_{model_config['learning_rate']}_"
        f"leaves_{model_config['num_leaves']}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "model": model_config["name"],
                "n_estimators": model_config["n_estimators"],
                "learning_rate": model_config["learning_rate"],
                "num_leaves": model_config["num_leaves"],
                "model_random_state": model_config["random_state"],
                "test_size": training_config["test_size"],
                "split_random_state": training_config["random_state"],
            }
        )

        mlflow.set_tags(
            {
                "git_commit": get_git_commit(),
                "data_md5": get_data_version(),
                "dataset": "Pima Indians Diabetes",
                "preprocessing": "train_only_imputation",
            }
        )

        mlflow.log_artifact(str(CONFIG_PATH))

        logger.info("Начато обучение LightGBM")

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))
            logger.info("%s: %.4f", name, value)

        print()
        print("Classification report:")
        print(classification_report(y_test, predictions))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODELS_DIR / "lightgbm_diabetes.joblib"
        scaler_path = MODELS_DIR / "scaler.joblib"
        imputer_path = MODELS_DIR / "imputer.joblib"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(imputer, imputer_path)

        # Сохраняем артефакты конкретного эксперимента.
        mlflow.log_artifact(str(model_path), "model")
        mlflow.log_artifact(str(scaler_path), "model")
        mlflow.log_artifact(str(imputer_path), "model")

        logger.info("Модель и предобработка сохранены")
        logger.info("MLflow Run ID: %s", run.info.run_id)

    logger.info("Обучение успешно завершено")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Ошибка при обучении модели")
        raise
