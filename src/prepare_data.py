from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/diabetes.csv")
OUTPUT_PATH = Path("data/processed/diabetes_processed.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Исходный датасет не найден: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Размер исходного датасета: {df.shape}")
    print()
    print("Первые 5 строк:")
    print(df.head())

    cols_with_zeros = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    df_processed = df.copy()

    # Физиологически невозможные нули считаем пропусками
    df_processed[cols_with_zeros] = (
        df_processed[cols_with_zeros]
        .replace(0, np.nan)
    )

    # Медианная импутация отдельно для каждого класса
    for col in cols_with_zeros:
        median_0 = (
            df_processed[
                df_processed["Outcome"] == 0
            ][col]
            .median()
        )

        median_1 = (
            df_processed[
                df_processed["Outcome"] == 1
            ][col]
            .median()
        )

        mask_0 = (
            (df_processed["Outcome"] == 0)
            & df_processed[col].isna()
        )

        mask_1 = (
            (df_processed["Outcome"] == 1)
            & df_processed[col].isna()
        )

        df_processed.loc[mask_0, col] = median_0
        df_processed.loc[mask_1, col] = median_1

    # Feature Engineering из курсовой
    df_processed["Insulin_BMI"] = (
        df_processed["Insulin"]
        * df_processed["BMI"]
    )

    df_processed["Glucose_BMI"] = (
        df_processed["Glucose"]
        * df_processed["BMI"]
    )

    df_processed["Glucose_Age"] = (
        df_processed["Glucose"]
        * df_processed["Age"]
    )

    df_processed["BMI_Age"] = (
        df_processed["BMI"]
        * df_processed["Age"]
    )

    df_processed["Insulin_Glucose"] = (
        df_processed["Insulin"]
        * df_processed["Glucose"]
    )

    df_processed["Pregnancies_Age"] = (
        df_processed["Pregnancies"]
        * df_processed["Age"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_processed.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Подготовка данных завершена")
    print(
        f"Пропусков после обработки: "
        f"{df_processed.isnull().sum().sum()}"
    )
    print(
        f"Обработанный датасет сохранён: "
        f"{OUTPUT_PATH}"
    )
    print(
        f"Размер обработанного датасета: "
        f"{df_processed.shape}"
    )


if __name__ == "__main__":
    main()