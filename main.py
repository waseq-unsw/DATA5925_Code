from helper_functions import add_weekday_weekend_flags, load_yjmob_dataset

# Set number of rows (for testing)
SAMPLE_ROWS = 100_000
DATASET_NAMES = [
    "yjmob100k-dataset1.csv.gz",
    "yjmob100k-dataset2.csv.gz",
]

def main() -> None:
    # Use nrows=None to load full datasets.
    for name in DATASET_NAMES:
        df = load_yjmob_dataset(name, nrows=SAMPLE_ROWS)
        day_encoded_df = add_weekday_weekend_flags(df)
        print(f"\n{name}")
        print(f"Rows: {len(day_encoded_df)}")
        print(
            "Distribution: "
            f"is_weekday={day_encoded_df['is_weekday'].sum()}, "
            f"is_weekend={day_encoded_df['is_weekend'].sum()}"
        )
        print(day_encoded_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
