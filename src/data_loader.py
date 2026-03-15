from pathlib import Path
import zipfile
import pandas as pd


def extract_zip_if_needed(zip_path: str | Path, extract_dir: str | Path) -> Path:
    """
    Распаковывает zip-архив, если папка extract_dir ещё не существует
    или пуста.

    Parameters
    ----------
    zip_path : str | Path
        Путь к zip-файлу.
    extract_dir : str | Path
        Папка, куда распаковывать архив.

    Returns
    -------
    Path
        Путь к папке с распакованными файлами.
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    if not any(extract_dir.iterdir()):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    return extract_dir


def read_tsf(file_path: str | Path) -> pd.DataFrame:
    """
    Читает .tsf файл и возвращает DataFrame с колонками:
    - series_name
    - start_timestamp
    - values

    Parameters
    ----------
    file_path : str | Path
        Путь к .tsf файлу.

    Returns
    -------
    pd.DataFrame
    """
    file_path = Path(file_path)

    data = []
    series_names = []
    start_timestamps = []

    with open(file_path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    data_section = False

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("@data"):
            data_section = True
            continue

        if data_section:
            parts = line.split(":")

            series_name = parts[0]
            start_timestamp = parts[1]
            values = parts[-1].split(",")

            values = [float(v) if v != "?" else None for v in values]

            series_names.append(series_name)
            start_timestamps.append(start_timestamp)
            data.append(values)

    df = pd.DataFrame(
        {
            "series_name": series_names,
            "start_timestamp": start_timestamps,
            "values": data,
        }
    )

    return df


def tsf_to_long_format(tsf_df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Преобразует DataFrame из TSF-формата в long format.

    Parameters
    ----------
    tsf_df : pd.DataFrame
        DataFrame с колонками series_name, start_timestamp, values.
    freq : str
        Частота временного ряда. Для M4 daily это 'D'.

    Returns
    -------
    pd.DataFrame
        DataFrame с колонками:
        - series_id
        - timestamp
        - value
    """
    long_rows = []

    for _, row in tsf_df.iterrows():
        start = pd.to_datetime(row["start_timestamp"])
        values = row["values"]

        dates = pd.date_range(
            start=start,
            periods=len(values),
            freq=freq,
        )

        temp_df = pd.DataFrame(
            {
                "series_id": row["series_name"],
                "timestamp": dates,
                "value": values,
            }
        )

        long_rows.append(temp_df)

    long_df = pd.concat(long_rows, ignore_index=True)
    return long_df


def sample_series(
    long_df: pd.DataFrame,
    n_series: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Сэмплирует n_series уникальных рядов из long_df.

    Parameters
    ----------
    long_df : pd.DataFrame
        DataFrame в long format.
    n_series : int
        Сколько рядов взять.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
    """
    unique_series = long_df["series_id"].drop_duplicates()

    if n_series > len(unique_series):
        raise ValueError(
            f"Запрошено {n_series} рядов, но доступно только {len(unique_series)}."
        )

    sample_ids = unique_series.sample(n=n_series, random_state=random_state)

    sample_df = long_df[long_df["series_id"].isin(sample_ids)].copy()
    return sample_df


def prepare_forecasting_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит DataFrame к формату:
    - unique_id
    - ds
    - y

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    result = df.rename(
        columns={
            "series_id": "unique_id",
            "timestamp": "ds",
            "value": "y",
        }
    ).copy()

    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return result


def load_m4_daily_sample(
    zip_path: str | Path = "data/m4_daily_dataset.zip",
    extract_dir: str | Path = "data/m4_daily",
    tsf_filename: str = "m4_daily_dataset.tsf",
    n_series: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Полный пайплайн загрузки M4 Daily:
    1. распаковать zip при необходимости
    2. прочитать tsf
    3. перевести в long format
    4. взять случайную выборку рядов
    5. привести к формату unique_id, ds, y

    Parameters
    ----------
    zip_path : str | Path
        Путь к zip-файлу.
    extract_dir : str | Path
        Папка для распаковки.
    tsf_filename : str
        Имя tsf-файла внутри распакованной папки.
    n_series : int
        Количество рядов в выборке.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Готовый DataFrame с колонками unique_id, ds, y.
    """
    extract_dir = extract_zip_if_needed(zip_path=zip_path, extract_dir=extract_dir)

    tsf_path = Path(extract_dir) / tsf_filename
    tsf_df = read_tsf(tsf_path)
    long_df = tsf_to_long_format(tsf_df, freq="D")
    sample_df = sample_series(long_df, n_series=n_series, random_state=random_state)
    final_df = prepare_forecasting_dataframe(sample_df)

    return final_df