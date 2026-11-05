import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.train import create_features, replace_invalid_zeros


def test_replace_invalid_zeros():
    """Проверяем замену некорректных нулей на NaN."""
    df = pd.DataFrame(
        {
            "Glucose": [120, 0],
            "BloodPressure": [80, 70],
            "SkinThickness": [0, 25],
            "Insulin": [100, 0],
            "BMI": [25.5, 30.0],
        }
    )

    result = replace_invalid_zeros(df)

    assert np.isnan(result.loc[1, "Glucose"])
    assert np.isnan(result.loc[0, "SkinThickness"])
    assert np.isnan(result.loc[1, "Insulin"])
    assert result.loc[0, "Glucose"] == 120
    assert df.loc[1, "Glucose"] == 0


def test_create_features():
    """Проверяем создание дополнительных признаков."""
    df = pd.DataFrame(
        {
            "Pregnancies": [2],
            "Glucose": [140],
            "BloodPressure": [80],
            "SkinThickness": [30],
            "Insulin": [120],
            "BMI": [31.5],
            "DiabetesPedigreeFunction": [0.5],
            "Age": [45],
        }
    )

    result = create_features(df)

    assert result["Glucose_BMI"].iloc[0] == 140 * 31.5
    assert result["Insulin_BMI"].iloc[0] == 120 * 31.5
    assert result["Glucose_Age"].iloc[0] == 140 * 45
    assert result["Pregnancies_Age"].iloc[0] == 2 * 45
    assert result.shape[1] == 14


def test_imputer():
    """Проверяем заполнение пропусков медианой train."""
    train = pd.DataFrame({"Glucose": [100.0, np.nan, 140.0]})
    test = pd.DataFrame({"Glucose": [np.nan, 200.0]})

    imputer = SimpleImputer(strategy="median")

    train_result = imputer.fit_transform(train)
    test_result = imputer.transform(test)

    assert train_result[1, 0] == 120.0
    assert test_result[0, 0] == 120.0
    assert test_result[1, 0] == 200.0


def test_prediction_format():
    """Проверяем формат предсказания обученной модели."""
    from src.predict import predict_patient

    patient = {
        "Pregnancies": 2,
        "Glucose": 140,
        "BloodPressure": 80,
        "SkinThickness": 30,
        "Insulin": 120,
        "BMI": 31.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45,
    }

    result = predict_patient(patient)

    assert set(result) == {
        "prediction",
        "risk",
        "probability",
    }
    assert result["prediction"] in (0, 1)
    assert result["risk"] in ("Диабет", "Нет диабета")
    assert 0.0 <= result["probability"] <= 1.0
