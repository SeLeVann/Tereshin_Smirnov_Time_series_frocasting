import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

from src.data_loader import load_m4_daily_sample
from src.catboost_model import train_test_split_by_horizon, run_catboost_experiment
from src.patchtst_model import run_patchtst_experiment
from src.baselines import run_baseline_experiment


# -----------------------
# параметры эксперимента
# -----------------------

h = 50
lags = list(range(1, 15)) + [21, 28, 35, 42, 49]

scalers = {
    "none": None,
    "standard": StandardScaler,
    "robust": RobustScaler,
    "quantile": lambda: QuantileTransformer(
        output_distribution="normal",
        n_quantiles=100,
        random_state=42
    )
}


# -----------------------
# загрузка данных
# -----------------------

print("Loading dataset...")
df = load_m4_daily_sample()

print("Splitting train/test...")
train, test = train_test_split_by_horizon(df, h=h)


# -----------------------
# запуск экспериментов
# -----------------------

results = []

# ---------- CatBoost ----------
for name, scaler in scalers.items():
    smape_val, mase_val = run_catboost_experiment(
        train=train,
        test=test,
        lags=lags,
        h=h,
        scaler_name=name,
        scaler_cls=scaler
    )

    results.append({
        "model": "CatBoost",
        "scaler": name,
        "sMAPE": smape_val,
        "MASE": mase_val
    })


# ---------- PatchTST ----------
for name, scaler in scalers.items():
    smape_val, mase_val = run_patchtst_experiment(
        train=train,
        test=test,
        h=h,
        scaler_name=name,
        scaler_cls=scaler
    )

    results.append({
        "model": "PatchTST",
        "scaler": name,
        "sMAPE": smape_val,
        "MASE": mase_val
    })


# ---------- Baselines ----------
baseline_models = ["Naive", "SeasonalNaive"]

for model_name in baseline_models:
    for name, scaler in scalers.items():
        smape_val, mase_val = run_baseline_experiment(
            train=train,
            test=test,
            h=h,
            scaler_name=name,
            scaler_cls=scaler,
            model_type=model_name
        )

        results.append({
            "model": model_name,
            "scaler": name,
            "sMAPE": smape_val,
            "MASE": mase_val
        })


# -----------------------
# результаты
# -----------------------

results_df = pd.DataFrame(results)

print("\nExperiment results:\n")
print(results_df)

# при желании можно сохранить
results_df.to_csv("results/experiment_results.csv", index=False)