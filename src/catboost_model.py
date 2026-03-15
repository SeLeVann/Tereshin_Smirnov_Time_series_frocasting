import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from catboost import CatBoostRegressor


def train_test_split_by_horizon(df: pd.DataFrame, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит каждый ряд на train/test:
    последние h точек идут в test, остальные в train.
    """
    train = df.groupby("unique_id").head(-h).copy()
    test = df.groupby("unique_id").tail(h).copy()
    return train, test


def smape(y, y_hat, eps=1e-8):
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    return np.mean(2 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat) + eps))


def mase(train_y, test_y, pred_y, seasonality=7, eps=1e-8):
    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)
    pred_y = np.asarray(pred_y)

    if len(train_y) <= seasonality:
        return np.nan

    naive_errors = np.abs(train_y[seasonality:] - train_y[:-seasonality])
    scale = np.mean(naive_errors)

    if scale < eps:
        return np.nan

    return np.mean(np.abs(test_y - pred_y)) / scale


def scale_series_fit(df: pd.DataFrame, scaler_cls):
    """
    Обучает отдельный scaler на каждом ряде train.
    """
    scaled_parts = []
    scalers = {}

    for uid, g in df.groupby("unique_id"):
        scaler = scaler_cls()
        vals = g["y"].values.reshape(-1, 1)

        g2 = g.copy()
        g2["y"] = scaler.fit_transform(vals).ravel()

        scaled_parts.append(g2)
        scalers[uid] = scaler

    return pd.concat(scaled_parts, ignore_index=True), scalers


def scale_series_transform(df: pd.DataFrame, scalers: dict):
    """
    Применяет уже обученные scaler'ы к test.
    """
    scaled_parts = []

    for uid, g in df.groupby("unique_id"):
        scaler = scalers[uid]

        vals = g["y"].values.reshape(-1, 1)
        g2 = g.copy()
        g2["y"] = scaler.transform(vals).ravel()

        scaled_parts.append(g2)

    return pd.concat(scaled_parts, ignore_index=True)


def make_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    Добавляет лаговые признаки внутри каждого ряда.
    """
    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("unique_id")["y"].shift(lag)

    return df


def recursive_forecast(model, train_scaled: pd.DataFrame, h: int, lags: list[int]) -> pd.DataFrame:
    """
    Рекурсивный прогноз на h шагов вперёд для каждого ряда.
    """
    preds = []

    for uid, g in train_scaled.groupby("unique_id"):
        history = list(g["y"].values)

        for step in range(h):
            features = [history[-lag] for lag in lags]
            X = np.array(features).reshape(1, -1)

            y_hat = model.predict(X)[0]

            preds.append(
                {
                    "unique_id": uid,
                    "step": step,
                    "y_hat": y_hat,
                }
            )

            history.append(y_hat)

    return pd.DataFrame(preds)


def get_scalers() -> dict:
    """
    Возвращает словарь scaler'ов для эксперимента.
    """
    return {
        "none": None,
        "standard": StandardScaler,
        "robust": RobustScaler,
        "quantile": lambda: QuantileTransformer(
            output_distribution="normal",
            n_quantiles=100,
            random_state=42,
        ),
    }


def run_catboost_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    lags: list[int],
    h: int,
    scaler_name: str,
    scaler_cls,
    seasonality: int = 7,
) -> tuple[float, float]:
    """
    Один запуск CatBoost для одного scaler'а.
    Возвращает средние sMAPE и MASE по рядам.
    """
    print(f"Running: {scaler_name}")

    # scaling
    if scaler_cls is not None:
        train_scaled, scalers = scale_series_fit(train, scaler_cls)
        test_scaled = scale_series_transform(test, scalers)
    else:
        train_scaled = train.copy()
        test_scaled = test.copy()

    # lag features
    train_lag = make_lags(train_scaled, lags).dropna()

    X_train = train_lag.drop(columns=["y", "unique_id", "ds"])
    y_train = train_lag["y"]

    # model
    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        verbose=50,
        random_seed=42,
    )

    model.fit(X_train, y_train)

    # prediction
    pred_df = recursive_forecast(model, train_scaled, h, lags)

    # prepare test
    test_df = test_scaled.copy()
    test_df["step"] = test_df.groupby("unique_id").cumcount()

    eval_df = test_df.merge(pred_df, on=["unique_id", "step"], how="left")

    # metrics per series
    smapes = []
    mases = []

    for uid, g in eval_df.groupby("unique_id"):
        train_y = train_scaled.loc[train_scaled["unique_id"] == uid, "y"].values
        test_y = g["y"].values
        pred_y = g["y_hat"].values

        smapes.append(smape(test_y, pred_y))
        mases.append(mase(train_y, test_y, pred_y, seasonality=seasonality))

    return float(np.nanmean(smapes)), float(np.nanmean(mases))


def run_all_catboost_experiments(
    df: pd.DataFrame,
    h: int = 50,
    lags = None,
    seasonality: int = 7,
) -> pd.DataFrame:
    """
    Полный прогон CatBoost по всем scaler'ам.
    На вход получает исходный df с колонками unique_id, ds, y.
    Возвращает таблицу результатов.
    """
    if lags is None:
        lags = list(range(1, 15)) + [21, 28, 35, 42, 49]

    train, test = train_test_split_by_horizon(df, h=h)
    scalers = get_scalers()

    results = []

    for name, scaler in scalers.items():
        smape_val, mase_val = run_catboost_experiment(
            train=train,
            test=test,
            lags=lags,
            h=h,
            scaler_name=name,
            scaler_cls=scaler,
            seasonality=seasonality,
        )

        results.append(
            {
                "model": "CatBoost",
                "scaler": name,
                "sMAPE": smape_val,
                "MASE": mase_val,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df