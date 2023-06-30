import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_prep import load_data
from utils.split import split_by_difference
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from utils.split import _fill_set


def filter_by_metric(args, return_distance_matrix: bool = False) -> pd.DataFrame:
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
    # Replace NaN with mean
    # mean_filler = selected_dataset.mean(numeric_only=True)
    selected_dataset = selected_dataset.fillna(selected_dataset.mean(numeric_only=True))
    # **This makes sure that each pair has the same distance:
    selected_dataset2 = deepcopy(selected_dataset)
    selected_dataset2.columns = [selected_dataset2.columns[1], selected_dataset2.columns[0], selected_dataset2.columns[2]]
    selected_dataset = pd.concat([selected_dataset, selected_dataset2], axis=0)
    if return_distance_matrix:
        selected_dataset = pd.pivot_table(
            selected_dataset, values=args.metric_to_split, index="seq2", columns="seq1"
        )
        # LEGACY - no longer True, see ** above:
            # No need to fill nan here because half of the table will have NaN
            # Also, the table won't be filled symmetrically, meaning filling with a value may affect the calculation of the mean/median
            # For any further purposes, it is not necessary to fill the matrix but a loop would be required to fill these nan values with the equivalent number present
            # at epitope1, epitope2 rather than epitope2, epitope1 which currently wmay give nan
    return selected_dataset


def get_dist(set_1, set_2, dist_array):
    dist_list = []
    # For each pair of epitopes:
    for m_1, m_2 in zip(set_1, set_2):
        # Find corresponding line with epitopes:
        curr_dist = dist_array[(dist_array["seq1"] == m_1) & (dist_array["seq2"] == m_2)]
        # Get last column (this may vary in name but will always be the last)
        dist_col = curr_dist.columns[-1]
        if len(curr_dist) > 0:
            # Select floating value of column and enforce conversion to float
            dist_list.append(float(curr_dist[dist_col].iloc[0]))
        else:
            dist_list.append(np.nan)
    return dist_list


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
        args.min_epitope_count <= args.max_epitope_count
    ), "Expected min_epitope_count to be smaller than max_epitope_count"
    # Select correct database and count:
    selected_dataset, epitopes_count_dict = load_data(args.dataset_path)
    assert len(args.lowest_percentiles) == len(
        args.highest_percentiles
    ), f"Lengths of low {len(args.lowest_percentiles)} and high {len(args.highest_percentiles)} are different"
    # Filter by metric:
    distance_matrix = filter_by_metric(
        args, return_distance_matrix=True
    )  # (n_epitope, n_epitope)
    distance_arr = filter_by_metric(args)  # Seq1, Seq2, Distance
    dist_mean = np.nanmean(distance_matrix, axis=1)
    dist_median = np.nanmedian(distance_matrix, axis=1)
    output_data = []
    for i, curr_epitope in enumerate(distance_matrix.columns):
        output_data.append(
            [
                curr_epitope,
                epitopes_count_dict[curr_epitope],
                dist_median[i],
                dist_mean[i],
            ]
        )
    df = pd.DataFrame(output_data, columns=["epitope", "count", "median", "mean"])
    df.to_csv(f"epitope_count_median_mean_{args.metric_to_split}.csv")
    # Split dataset:
    TV_dist, TT_dist = [], []
    TV_labels, TT_labels = [], []
    for l, h in zip(args.lowest_percentiles, args.highest_percentiles):
        l = float(l)
        h = float(h)
        assert (
            0 <= l < h <= 100
        ), f"Expected lowest_percentile value to be below highest_percentile but got low {l} and high {h}"
        l_p, h_p = np.percentile(dist_median, [l, h])
        epitopes_mask = (l_p <= dist_median) & (dist_median <= h_p)
        in_epitopes = distance_matrix.columns[epitopes_mask]
        out_epitopes = distance_matrix.columns[~epitopes_mask]
        # Calculate test budget by summing all and multiplying by cutoff:
        # Load dictionary to numpy array:
        epitopes_count_arr = np.array(list(epitopes_count_dict.items()))
        # Find inter-quartile range that contains the highest represented % of the data:
        counts = np.array(epitopes_count_arr[:, 1], dtype=int)
        min_split_count = int(counts.sum() * (1 - args.split_ratio)) // 2
        try:
            # Create test set with minimum distance:
            test_peptides_set, test_len = _fill_set(
                min_split_count,
                in_epitopes,
                epitopes_count_dict,
                leeway_val=args.leeway,
                min_epitope_count=args.min_epitope_count,
                max_epitope_count=args.max_epitope_count,
            )
            # Remaining peptides are train+val
            discarded_set = set(in_epitopes) - test_peptides_set
            train_val_peptides_set = set(out_epitopes).union(discarded_set)
            # Hard split on val:
            val_peptides_set, val_len = _fill_set(
                min_split_count,
                np.array(list(train_val_peptides_set)),
                epitopes_count_dict,
                leeway_val=args.leeway,
                min_epitope_count=args.min_epitope_count,
                max_epitope_count=args.max_epitope_count,
            )
            train_peptides_set = train_val_peptides_set - val_peptides_set
            TT = get_dist(train_peptides_set, test_peptides_set, distance_arr)
            TV = get_dist(train_peptides_set, val_peptides_set, distance_arr)
            TT_dist += TT
            TV_dist += TV
            TT_labels += [f"({l}, {h})"] * len(TT)
            TV_labels += [f"({l}, {h})"] * len(TV)
            # Export
            file_name = f"{args.dataset_path.stem}_diff_{args.metric_to_split}_split_{args.split_ratio}_minepitope_{args.min_epitope_count}_maxexpitope_{args.max_epitope_count}_min_{l}_max_{h}_leeway_{args.leeway}_{args.seed}.csv"
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
        except Exception as e:
            print(f"Unable to fill set with low {l} and high {h}")
            print(e)
    df_TV = pd.DataFrame(data={"Distance": TV_dist, "Percentile interval": TV_labels})
    df_TT = pd.DataFrame(data={"Distance": TT_dist, "Percentile interval": TT_labels})
    p = sns.displot(df_TV, x="Distance", hue="Percentile interval", multiple="stack")
    p.savefig(args.tv_fig_out)
    plt.clf()

    p = sns.displot(df_TT, x="Distance", hue="Percentile interval", multiple="stack")
    p.savefig(args.tt_fig_out)
    plt.clf()

    # Violin Plot
    p = sns.violinplot(y="Distance", x="Percentile interval", data=df_TV)
    p = p.get_figure()
    p.savefig(args.tv_fig_out + "_violin")
    plt.clf()

    p = sns.violinplot(y="Distance", x="Percentile interval", data=df_TT)
    p = p.get_figure()
    p.savefig(args.tt_fig_out + "_violin")
    plt.clf()
    df_TT.to_csv("df_tt.csv")
    df_TV.to_csv("df_tv.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to input file")
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
        "--min_epitope_count",
        type=int,
        default=50,
        help="Minimum count of epitopes to be included (default: 50)",
    )
    parser.add_argument(
        "--max_epitope_count",
        type=int,
        default=500,
        help="Maximum count of epitopes to be included (default: 50)",
    )
    parser.add_argument(
        "--lowest_percentiles",
        nargs="+",
        help="Lowest percentiles to select by similarity (default: 80)",
        required=True,
    )
    parser.add_argument(
        "--highest_percentiles",
        nargs="+",
        help="Lowest percentiles to select by similarity (default: 80)",
        required=True,
    )
    parser.add_argument(
        "--leeway",
        type=float,
        default=0.05,
        help="Sets will have around leeway % above and below the count (default: 0.05)",
    )
    parser.add_argument("--tt_fig_out", required=True)
    parser.add_argument("--tv_fig_out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    params = parser.parse_args()
    main(params)
