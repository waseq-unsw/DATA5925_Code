from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "10836269"
TARGET_FILES = [
    "yjmob100k-dataset1.csv.gz",
    "yjmob100k-dataset2.csv.gz",
]
OUTPUT_PLOT = Path(__file__).resolve().parent / "day_count_activity.png"
CHUNK_SIZE = 2_000_000


def count_records_by_day(file_path: Path, chunksize: int = CHUNK_SIZE) -> pd.Series:
    """Read a mobility file in chunks and return record counts by day index d."""
    day_counts: pd.Series | None = None
    for chunk in pd.read_csv(file_path, usecols=["d"], chunksize=chunksize):
        current = chunk["d"].value_counts().sort_index()
        if day_counts is None:
            day_counts = current
        else:
            day_counts = day_counts.add(current, fill_value=0)

    if day_counts is None:
        return pd.Series(dtype="int64")
    return day_counts.sort_index().astype("int64")


def infer_weekend_by_activity(day_counts: pd.Series) -> tuple[list[int], pd.Series]:
    """
    Infer likely weekend modulo classes from lower activity.

    Approach:
    - Group day counts by day_index % 7
    - Compute median activity per modulo class
    - Pick the 2 classes with lowest medians as likely weekend classes
    """
    by_mod = day_counts.groupby(day_counts.index % 7).median().sort_values()
    weekend_mod_classes = sorted(by_mod.index[:2].tolist())
    weekend_days = [int(day) for day in day_counts.index if day % 7 in weekend_mod_classes]
    return weekend_days, by_mod


def plot_day_counts(
    day_counts_by_dataset: dict[str, pd.Series],
    weekend_guess_by_dataset: dict[str, list[int]],
    output_path: Path,
) -> None:
    """Plot daily activity counts and highlight inferred weekend days."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    for ax, (name, counts) in zip(axes, day_counts_by_dataset.items()):
        ax.plot(counts.index, counts.values, marker="o", linewidth=1.2, markersize=3)
        ax.set_title(f"{name}: record count by day (d)")
        ax.set_xlabel("Day index (d)")
        ax.set_ylabel("Record count")
        ax.grid(alpha=0.3)

        weekend_days = weekend_guess_by_dataset.get(name, [])
        weekend_points = counts[counts.index.isin(weekend_days)]
        if not weekend_points.empty:
            ax.scatter(
                weekend_points.index,
                weekend_points.values,
                color="tab:red",
                s=25,
                label="inferred weekend",
                zorder=3,
            )
            ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    day_counts_by_dataset: dict[str, pd.Series] = {}
    weekend_guess_by_dataset: dict[str, list[int]] = {}

    print(f"Reading files from: {DATA_DIR}")
    for filename in TARGET_FILES:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            print(f"Missing file: {file_path}")
            continue

        print(f"\nProcessing {filename} ...")
        counts = count_records_by_day(file_path)
        if counts.empty:
            print("No rows found.")
            continue

        day_counts_by_dataset[filename] = counts
        weekend_days, mod_medians = infer_weekend_by_activity(counts)
        weekend_guess_by_dataset[filename] = weekend_days

        print(f"Distinct d values ({len(counts.index)}): {counts.index.tolist()}")
        print("Counts by d:")
        print(counts.to_string())
        print("\nMedian activity by d % 7:")
        print(mod_medians.to_string())
        print(
            "Inferred weekend modulo classes (lowest 2 medians): "
            f"{sorted(mod_medians.index[:2].tolist())}"
        )
        print(f"Inferred weekend d values: {weekend_days}")

    if not day_counts_by_dataset:
        print("No valid datasets processed; skipping plot.")
        return

    plot_day_counts(day_counts_by_dataset, weekend_guess_by_dataset, OUTPUT_PLOT)
    print(f"\nSaved plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
