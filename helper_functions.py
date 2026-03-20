import pandas as pd

WEEKEND_LIST = [0, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 41, 42, 48, 49, 55, 56, 62, 63, 69, 70]


def load_yjmob_dataset(filename, nrows):
    """
    Load one yjmob dataset file from data/.

    Parameters:
    - filename: dataset filename (e.g. yjmob100k-dataset1.csv.gz)
    - nrows: optional row limit for faster loading
    """
    base_dir = "data"
    file_path = f"{base_dir}/{filename}"
    try:
        return pd.read_csv(file_path, nrows=nrows)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing dataset file: {file_path}")


def add_weekday_weekend_flags(
    df: pd.DataFrame,
    weekend_list: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add is_weekend and is_weekday columns based on d values.

    Rule:
    - if d is in weekend_list -> is_weekend=1, is_weekday=0
    - otherwise -> is_weekend=0, is_weekday=1
    """
    if "d" not in df.columns:
        raise KeyError("Column 'd' not found in dataframe.")

    weekends = set(weekend_list or WEEKEND_LIST)
    result = df.copy()
    result["is_weekend"] = result["d"].isin(weekends).astype(int)
    result["is_weekday"] = (1 - result["is_weekend"]).astype(int)
    return result
