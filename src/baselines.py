import numpy as np
import pandas as pd

from src.catboost_model import (
    smape,
    mase,
    scale_series_fit,
    scale_series_transform,
)


def naive_forecast(train: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Naive forecast:
    на всех h шагах прогноз равен последнему значению train.
    """
    preds = []

    for uid, g in train.groupby("unique_id"):
        last = g["y"].iloc[-1]

        for step in range(h):
            preds.append(
                {
                    "unique_id": uid,
                    "step": step,
                    "y_hat": last,
                }
            )

    return pd.DataFrame(preds)


def seasonal_naive_forecast(
    train: pd.DataFrame,
    h: int,
    seasonality: int = 7,
) -> pd.DataFrame:
    """
    Seasonal Naive forecast:
    прогноз берётся из последних seasonality наблюдений по циклу.
    """
    preds = []

    for uid, g in train.groupby("unique_id"):
        values = g["y"].values

        if len(values) < seasonality:
            raise ValueError(
                f"Ряд {uid} слишком короткий для seasonality={seasonality}"
            )

        seasonal_tail = values[-seasonality:]

        for step in range(h):
            y_hat = seasonal_tail[step % seasonality]

            preds.append(
                {
                    "unique_id": uid,
                    "step": step,
                    "y_hat": y_hat,
                }
            )

    return pd.DataFrame(preds)


def run_baseline_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    h: int,
    scaler_name: str,
    scaler_cls,
    model_type: str,
    seasonality: int = 7,
) -> tuple[float, float]:
    """
    Один запуск baseline-модели для одного scaler'а.

    model_type:
    - "Naive"
    - "SeasonalNaive"
    """
    print(f"Running baseline: {model_type}, scaler={scaler_name}")

    # scaling
    if scaler_cls is not None:
        train_scaled, scalers = scale_series_fit(train, scaler_cls)
        test_scaled = scale_series_transform(test, scalers)
    else:
        train_scaled = train.copy()
        test_scaled = test.copy()

    # forecast
    if model_type == "Naive":
        pred_df = naive_forecast(train_scaled, h)

    elif model_type == "SeasonalNaive":
        pred_df = seasonal_naive_forecast(
            train_scaled,
            h=h,
            seasonality=seasonality,
        )

    else:
        raise ValueError(f"Неизвестный baseline model_type: {model_type}")

    # prepare test
    test_df = test_scaled.copy()
    test_df["step"] = test_df.groupby("unique_id").cumcount()

    eval_df = test_df.merge(pred_df, on=["unique_id", "step"], how="left")

    # metrics
    smapes = []
    mases = []

    for uid, g in eval_df.groupby("unique_id"):
        train_y = train_scaled.loc[train_scaled["unique_id"] == uid, "y"].values
        test_y = g["y"].values
        pred_y = g["y_hat"].values

        smapes.append(smape(test_y, pred_y))
        mases.append(mase(train_y, test_y, pred_y, seasonality=seasonality))

    return float(np.nanmean(smapes)), float(np.nanmean(mases))