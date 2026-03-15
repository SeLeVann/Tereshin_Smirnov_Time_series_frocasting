import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

# импортируем метрики и scaling из catboost файла
from src.catboost_model import (
    smape,
    mase,
    scale_series_fit,
    scale_series_transform
)


def run_patchtst_experiment(train, test, h, scaler_name, scaler_cls):

    print("Running PatchTST:", scaler_name)

    # scaling

    if scaler_cls is not None:
        train_scaled, scalers = scale_series_fit(train, scaler_cls)
        test_scaled = scale_series_transform(test, scalers)
    else:
        train_scaled = train.copy()
        test_scaled = test.copy()

    # модель
    model = PatchTST(
        h=h,
        input_size=128,
        patch_len=16,
        stride=8,
        max_steps=100,
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    nf = NeuralForecast(
        models=[model],
        freq="D"
    )


    # обучение
    nf.fit(train_scaled)

    # prediction
    preds = nf.predict()

    preds = preds.rename(columns={"PatchTST": "y_hat"})

    # prepare test
    test_df = test_scaled.copy()
    test_df["step"] = test_df.groupby("unique_id").cumcount()

    preds["step"] = preds.groupby("unique_id").cumcount()

    eval_df = test_df.merge(preds, on=["unique_id", "step"])

    # метрики
    smapes = []
    mases = []

    for uid, g in eval_df.groupby("unique_id"):

        train_y = train_scaled.loc[
            train_scaled["unique_id"] == uid, "y"
        ].values

        test_y = g["y"].values
        pred_y = g["y_hat"].values

        smapes.append(smape(test_y, pred_y))
        mases.append(mase(train_y, test_y, pred_y, seasonality=7))

    return np.nanmean(smapes), np.nanmean(mases)