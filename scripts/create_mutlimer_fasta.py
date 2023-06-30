import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_prep import load_data
from utils.split import split_by_difference, split_by_hard



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
        0 < args.cutoff_split < 1
    ), f"Expected cutoff value to be below 1 but got {args.cutoff_split}"
    assert (
        0 < args.lowest_percentile < 100
    ), f"Expected lowest_percentile value to be below 100 but got {args.cutoff_split}"
    # Select correct database and count:
    selected_dataset, epitopes_count_dict = load_data(args.dataset_path)
    # Split dataset:
    if args.input_type == "difference":
        # Filter by metric:
        selected_dataset = filter_by_metric(args)
        train_peptides_set, test_peptides_set, val_peptides_set = split_by_difference(
            selected_dataset,
            split_ratio=args.cutoff_split,
            epitopes_count_dict=epitopes_count_dict,
            lowest_percentile=args.lowest_percentile,
        )
    elif args.input_type == "hard":
        train_peptides_set, test_peptides_set, val_peptides_set = split_by_hard(
            split_ratio=args.cutoff_split,
            epitopes_count_dict=epitopes_count_dict,
        )
    else:
        raise NotImplementedError
    file_name = f"{args.input_path.stem}-{args.input_type}-{args.metric_to_split}-{args.cutoff_split}-{args.seed}-{args.lowest_percentile}.csv"
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
    parser.add_argument("--seed", type=int, default=42)
    params = parser.parse_args()
    main(params)
