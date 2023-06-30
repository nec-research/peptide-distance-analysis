import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_prep import load_data
from utils.split import split_by_difference, split_by_hard


def filter_by_metric(args):
    metrics_df = pd.read_csv(args.input_path, skipinitialspace=True)
    # Ensure that column is in the metrics of the dataset
    assert (
        args.metric_to_split in metrics_df.columns
    ), f"{args.metric_to_split} not found in cols {metrics_df.columns}"
    # Find column for metric to split
    idx_metric = metrics_df.columns.tolist().index(args.metric_to_split)
    selected_dataset = metrics_df[metrics_df.columns[[0, 1, idx_metric]]]
    # Shuffle rows:
    selected_dataset = selected_dataset.sample(frac=1).reset_index(drop=True)
    return selected_dataset


def main(args):
    # Set random seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.input_path = Path(args.input_path)
    args.dataset_path = Path(args.dataset_path)
    # Sanity Checks
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    assert (
        args.dataset_path.exists()
    ), f"Dataset file {args.dataset_path} does not exist"
    assert (
        0 < args.split_ratio < 1
    ), f"Expected Split Ratio value to be below 1 but got {args.split_ratio}"
    assert (
        0 < args.lowest_percentile < args.highest_percentile <= 100
    ), f"Expected lowest_percentile value to be below highest_percentile but got low {args.lowest_percentile} and high {args.highest_percentile}"
    # Select correct database and count:
    selected_dataset, epitopes_count_dict = load_data(args.dataset_path)
    # Split dataset:
    if args.input_type == "difference":
        # Filter by metric:
        selected_dataset = filter_by_metric(args)
        train_peptides_set, test_peptides_set, val_peptides_set = split_by_difference(
            selected_dataset,
            split_ratio=args.split_ratio,
            epitopes_count_dict=epitopes_count_dict,
            lowest_percentile=args.lowest_percentile,
            highest_percentile=args.highest_percentile
        )
    elif args.input_type == "hard":
        train_peptides_set, test_peptides_set, val_peptides_set = split_by_hard(
            split_ratio=args.split_ratio,
            epitopes_count_dict=epitopes_count_dict,
        )
    else:
        raise NotImplementedError
    file_name = f"{args.input_path.stem}-{args.input_type}-{args.metric_to_split}-{args.split_ratio}-{args.seed}-low_{args.lowest_percentile}-high_{args.highest_percentile}.csv"
    np.savetxt(
        "train_" + file_name,
        np.array(list(train_peptides_set)),
        delimiter=",",
        fmt="%s",
    )
    np.savetxt(
        "test_" + file_name,
        np.array(list(test_peptides_set)),
        delimiter=",",
        fmt="%s",
    )
    np.savetxt(
        "val_" + file_name,
        np.array(list(val_peptides_set)),
        delimiter=",",
        fmt="%s",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to input file")
    parser.add_argument(
        "--input_type",
        default="abs_values",
        const="abs_values",
        nargs="?",
        choices=["difference", "hard"],
        help="Type of metrics to be used. Hard does not use any metrics",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to dataset file. Required for counts-based hardsplit.",
    )
    parser.add_argument(
        "--metric_to_split",
        type=str,
        nargs="?",
        help="Metric to split by. Must be in the title of the columns",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Ratio number to use for the split (default: 0.8)",
    )
    parser.add_argument(
        "--lowest_percentile",
        type=float,
        default=80,
        help="Lowest percentile to select by similarity (default: 80)",
    )
    parser.add_argument(
        "--highest_percentile",
        type=float,
        default=100,
        help="Highest percentile to select by similarity (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=42)
    params = parser.parse_args()
    main(params)
