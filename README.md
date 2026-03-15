# Tereshin_Smirnov_Time_series_frocasting

## Структура проекта

- **src/** — основной код проекта  
  - `baselines.py` — базовые модели (Naive, Seasonal Naive)  
  - `catboost_model.py` — модель CatBoost для прогнозирования  
  - `patchtst_model.py` — нейросетевая модель PatchTST  
  - `data_loader.py` — загрузка и подготовка датасета  

- **data/** — данные проекта  
  - `m4_daily/` — распакованный датасет M4  
  - `m4_daily_dataset.zip` — архив с исходным датасетом  

- **results/** — результаты экспериментов  
  - `experiment_results.csv` — таблица метрик моделей  

- **run_experiment.py** — основной скрипт запуска экспериментов  

- **config.py** — параметры эксперимента  

- **requirements.txt** — список зависимостей проекта  

- **catboost_info/** — автоматически создаваемые логи обучения CatBoost  

- **lightning_logs/** — автоматически создаваемые логи обучения PatchTST
