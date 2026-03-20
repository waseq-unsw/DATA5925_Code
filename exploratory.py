from pathlib import Path

import pandas as pd


DEFAULT_SAMPLE_ROWS = 100_000
DEFAULT_SAMPLE_ONLY = True
DEFAULT_PREVIEW_ROWS = 5


def get_data_dir() -> Path:
    """Return the absolute path to the project-level data folder."""
    return Path(__file__).resolve().parents[1] / "data"


def load_data_files(
    data_dir: Path,
    sample_only: bool = DEFAULT_SAMPLE_ONLY,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> dict[str, pd.DataFrame]:
    """Load CSV/GZ files under data directory, optionally as samples."""
    csv_files = set(data_dir.rglob("*.csv"))
    gz_files = set(data_dir.rglob("*.csv.gz")) | set(data_dir.rglob("*.gz"))
    all_files = sorted(csv_files | gz_files)
    if not all_files:
        raise FileNotFoundError(f"No CSV or GZ files found in: {data_dir}")

    datasets: dict[str, pd.DataFrame] = {}
    nrows = sample_rows if sample_only else None
    for file_path in all_files:
        relative_name = str(file_path.relative_to(data_dir))
        datasets[relative_name] = pd.read_csv(file_path, nrows=nrows)
    return datasets


def infer_timestamp_columns(df: pd.DataFrame) -> list[str]:
    """Infer likely timestamp columns from dataframe column names."""
    timestamp_keywords = ("time", "timestamp", "date", "datetime")
    exact_timestamp_names = {"t", "ts", "dt"}
    return [
        col
        for col in df.columns
        if col.lower() in exact_timestamp_names
        or any(keyword in col.lower() for keyword in timestamp_keywords)
    ]


def encode_weekday_weekend(
    df: pd.DataFrame,
    source_col: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Add weekday/weekend encodings from a day-index or timestamp column.

    Creates:
    - <prefix>_day_name: Mon, Tue, ...
    - <prefix>_is_weekend: 1 for Sat/Sun, else 0
    - <prefix>_day_type: 'weekday' / 'weekend'
    """
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in dataframe.")

    col_prefix = prefix or source_col

    # Prefer day-index interpretation for integer-like columns (e.g., d=0,1,2...).
    numeric_source = pd.to_numeric(df[source_col], errors="coerce")
    if numeric_source.notna().any() and numeric_source.dropna().max() < 100_000:
        day_of_week = (numeric_source % 7).astype("Int64")
        day_name_map = {
            0: "Mon",
            1: "Tue",
            2: "Wed",
            3: "Thu",
            4: "Fri",
            5: "Sat",
            6: "Sun",
        }
        df[f"{col_prefix}_day_name"] = day_of_week.map(day_name_map)
        df[f"{col_prefix}_is_weekend"] = day_of_week.isin([5, 6]).astype(int)
    else:
        converted = pd.to_datetime(df[source_col], errors="coerce")
        invalid_count = int(converted.isna().sum())
        if invalid_count == len(df):
            raise ValueError(
                f"Column '{source_col}' is neither a valid day index nor parseable datetime."
            )

        # Monday=0 ... Sunday=6. Weekend is Saturday/Sunday.
        day_of_week = converted.dt.dayofweek
        is_weekend = day_of_week >= 5
        df[f"{col_prefix}_day_name"] = converted.dt.day_name().str[:3]
        df[f"{col_prefix}_is_weekend"] = is_weekend.fillna(False).astype(int)

    df[f"{col_prefix}_day_type"] = df[f"{col_prefix}_is_weekend"].map(
        {0: "weekday", 1: "weekend"}
    )

    return df


def analyse_timestamp_distribution(df: pd.DataFrame, day_type_col: str) -> pd.Series:
    """Return weekday/weekend counts for quick analysis."""
    return df[day_type_col].value_counts(dropna=False)


def explain_columns(df: pd.DataFrame) -> list[str]:
    """Return simple, human-readable descriptions for common columns."""
    descriptions: list[str] = []
    known_meanings = {
        "uid": "user id (identifier for a person/device)",
        "d": "day index (integer day counter, e.g. 0, 1, 2...)",
        "t": "time slot within a day (often hour-like index)",
        "x": "grid x-coordinate (spatial location)",
        "y": "grid y-coordinate (spatial location)",
        "POIcategory": "point-of-interest category id",
        "category": "point-of-interest category id",
        "POI_count": "number of POIs for that grid cell/category row",
    }

    for col in df.columns:
        meaning = known_meanings.get(col, "no predefined meaning in this script")
        descriptions.append(f"{col}: {meaning}")
    return descriptions


def numeric_column_stats(df: pd.DataFrame, col: str) -> str:
    """Return count/range info for numeric-like columns."""
    numeric = pd.to_numeric(df[col], errors="coerce")
    if numeric.notna().sum() == 0:
        return "no numeric values"
    return (
        f"unique={int(numeric.nunique(dropna=True))}, "
        f"min={numeric.min():.0f}, max={numeric.max():.0f}"
    )


def describe_index_column(
    df: pd.DataFrame,
    col: str,
    max_show_distinct: int = 120,
    top_n_distribution: int = 15,
) -> list[str]:
    """Return detailed distribution notes for index-like columns such as d/t."""
    numeric = pd.to_numeric(df[col], errors="coerce")
    valid = numeric.dropna().astype(int)
    if valid.empty:
        return [f"{col}: no numeric values"]

    distinct_sorted = sorted(valid.unique().tolist())
    distinct_count = len(distinct_sorted)
    value_counts = valid.value_counts(dropna=False).sort_index()
    distribution_pct = (value_counts / len(valid) * 100).round(2)

    lines = [
        (
            f"{col}: unique={distinct_count}, min={valid.min()}, max={valid.max()}, "
            f"records={len(valid)}"
        )
    ]
    if distinct_count <= max_show_distinct:
        lines.append(f"{col} distinct values: {distinct_sorted}")
    else:
        preview_n = min(15, distinct_count)
        lines.append(
            f"{col} distinct values: first {preview_n}={distinct_sorted[:preview_n]} "
            f"... last {preview_n}={distinct_sorted[-preview_n:]}"
        )

    lines.append(f"{col} frequency distribution (value: count, percent):")
    if len(value_counts) <= top_n_distribution:
        for idx, count in value_counts.items():
            lines.append(f"  {idx}: {count}, {distribution_pct.loc[idx]}%")
    else:
        lines.append(f"  showing top {top_n_distribution} by count:")
        top_counts = valid.value_counts().head(top_n_distribution)
        top_pct = (top_counts / len(valid) * 100).round(2)
        for idx, count in top_counts.items():
            lines.append(f"  {int(idx)}: {int(count)}, {top_pct.loc[idx]}%")

    return lines


if __name__ == "__main__":
    data_dir = get_data_dir()
    print(f"Reading data from: {data_dir}")
    print(
        f"Sample mode: {DEFAULT_SAMPLE_ONLY} "
        f"(rows per file: {DEFAULT_SAMPLE_ROWS if DEFAULT_SAMPLE_ONLY else 'all'})"
    )

    dataframes = load_data_files(
        data_dir,
        sample_only=DEFAULT_SAMPLE_ONLY,
        sample_rows=DEFAULT_SAMPLE_ROWS,
    )
    print(f"Loaded {len(dataframes)} CSV/GZ file(s):")
    for name, df in dataframes.items():
        print(f" - {name}: {df.shape[0]} rows (records) x {df.shape[1]} columns")
        print("   Column meanings:")
        for line in explain_columns(df):
            print(f"   - {line}")
        if "d" in df.columns:
            print(
                "   d summary (day index): "
                f"{numeric_column_stats(df, 'd')} (this is not a row count)"
            )
            for line in describe_index_column(df, "d"):
                print(f"   {line}")
        if "t" in df.columns:
            print(
                "   t summary (time slot): "
                f"{numeric_column_stats(df, 't')} (this is not a row count)"
            )
            for line in describe_index_column(df, "t"):
                print(f"   {line}")
        print(f"   Preview (first {DEFAULT_PREVIEW_ROWS} rows):")
        print(df.head(DEFAULT_PREVIEW_ROWS).to_string(index=False))
        timestamp_cols = infer_timestamp_columns(df)
        day_index_col = "d" if "d" in df.columns else None
        if not timestamp_cols and not day_index_col:
            print("   No timestamp-like columns detected.")
            print()
            continue

        if day_index_col:
            try:
                encode_weekday_weekend(df, day_index_col, prefix="day_index")
                day_type_col = "day_index_day_type"
                distribution = analyse_timestamp_distribution(df, day_type_col)
                print(f"   Encoded '{day_index_col}' -> {day_type_col}")
                print("   Weekday/weekend counts below are record counts (number of rows):")
                print(f"{distribution.to_string()}\n")
            except (ValueError, KeyError) as err:
                print(f"   Skipped '{day_index_col}': {err}")

        for timestamp_col in timestamp_cols:
            if day_index_col and timestamp_col == day_index_col:
                continue
            try:
                encode_weekday_weekend(df, timestamp_col)
                day_type_col = f"{timestamp_col}_day_type"
                distribution = analyse_timestamp_distribution(df, day_type_col)
                print(f"   Encoded '{timestamp_col}' -> {day_type_col}")
                print("   Weekday/weekend counts below are record counts (number of rows):")
                print(f"{distribution.to_string()}\n")
            except (ValueError, KeyError) as err:
                print(f"   Skipped '{timestamp_col}': {err}")
