import typing as t

import numpy as np
import pandas as pd


def _fill_set(
    min_test_count: int,
    epitopes: np.ndarray,
    epitopes_count_dict: t.Dict,
    leeway_val: float = 0.05,
    min_epitope_count: int = 50,
    max_epitope_count: int = 500,
) -> t.Tuple[t.Set, int]:
    """
    Fills a split of the data by random sampling and ensuring the number of pairs
    of the epitope is within distance.

    The split tries to get as close to the n_split quota as possible, giving a max
    of 5% leeway.

    Parameters
    ----------
    min_test_count: int
        Number of total pairs to add in the split.
    epitopes: np.ndarray
        List of epitopes sequences
    epitopes_count_dict: dict
        {epitope_seq: pairs_count}

    Returns
    -------
    selected_epitopes: set
        Set of selected epitope sequences
    curr_split_len: int
        Number of pairs in the split
    """
    curr_split_len = 0
    selected_epitopes = []
    discarded_epitopes = []
    assert (
        0 <= leeway_val < 1.0
    ), "Leeway Value should be a percentage value between 0 or 1"
    while curr_split_len < min_test_count * (1-leeway_val):
        curr_epitope_idx = np.random.randint(0, epitopes.size)
        curr_epitope = epitopes[curr_epitope_idx]
        # Remove epitope from possible list:
        epitopes = np.delete(epitopes, curr_epitope_idx)
        # If epitope count is out of bounds, remove.
        if min_epitope_count <= epitopes_count_dict[curr_epitope] <= max_epitope_count:
            # Calculate theoretical count if epitope is added:
            curr_count = curr_split_len + epitopes_count_dict[curr_epitope]
            # If the count is within 5% of the wanted % of the data split:
            if curr_count/min_test_count < ( 1 + leeway_val):
                # Append Epitopes + count
                selected_epitopes.append(curr_epitope)
                curr_split_len += epitopes_count_dict[curr_epitope]
            else:
                # Discard the epitope:
                discarded_epitopes.append(curr_epitope)
        else:
            # Discard the epitope:
            discarded_epitopes.append(curr_epitope)
        # Closing conditions:
        if len(selected_epitopes) + len(discarded_epitopes) == len(
            epitopes_count_dict.keys()
        ):
            raise Exception("No more values to sample from.")
        if epitopes.size == 0:
            raise Exception(
                "Reached end of epitopes. No more values to sample from - relax your boundaries."
            )
    return set(selected_epitopes), curr_split_len


def _fill_set_similarity(
    min_test_count: int,
    epitopes: np.ndarray,
    epitopes_count_dict: t.Dict,
    epitopes_within_range_df: pd.DataFrame,
    leeway_val: float = 0.0,
) -> (t.Set, t.Set, int):
    """
    Fills a split of the data by random sampling and ensuring the of pairs
    of the epitope is within distance.

    The split tries to get as close to the n_split quota as possible, giving a max
    of 5% leeway.

    Parameters
    ----------
    min_test_count: int
        Number of total pairs to add in the split.
    epitopes: np.ndarray
        List of epitopes sequences
    epitopes_count_dict: dict
        {epitope_seq: pairs_count}

    Returns
    -------
    selected_epitopes: set
        Set of selected epitope sequences
    curr_split_len: int
        Number of pairs in the split
    """
    print(f"Starting with {len(epitopes)} epitopes")
    curr_split_len = 0
    split_selected_epitopes = []
    train_selected_epitopes = []
    discarded_epitopes = []
    # Sort due to sets being random:
    epitopes.sort()
    while curr_split_len < min_test_count and epitopes.size:
        curr_epitope_idx = np.random.randint(0, epitopes.size)
        curr_epitope = epitopes[curr_epitope_idx]
        if curr_epitope not in split_selected_epitopes:
            # Calculate theoretical count if epitope is added:
            curr_count = curr_split_len + epitopes_count_dict[curr_epitope]
            # If the count is within leeway_val% of the wanted % of the data split:
            if curr_count < (min_test_count + min_test_count * leeway_val):
                # Append Epitopes + count
                split_selected_epitopes.append(curr_epitope)
                curr_split_len += epitopes_count_dict[curr_epitope]
                # Find all distances including the current epitope (either in seq1 or in seq2)
                nearby_epitopes_df = epitopes_within_range_df[
                    (epitopes_within_range_df["seq1"] == curr_epitope)
                    | (epitopes_within_range_df["seq2"] == curr_epitope)
                ]
                # Of the nearby epitopes, remove those unavailable
                nearby_epitopes = list(nearby_epitopes_df["seq1"].unique()) + list(
                    nearby_epitopes_df["seq2"].unique()
                )
                # Remove selected epitope from possible list:
                epitopes = np.delete(epitopes, curr_epitope_idx)
                # Remove current epitope:
                nearby_epitopes_set = set(nearby_epitopes) - set(curr_epitope)
                # Remove selected or discarded epitopes
                nearby_epitopes_set -= set(split_selected_epitopes)
                nearby_epitopes_set -= set(discarded_epitopes)
                nearby_epitopes_set -= set(train_selected_epitopes)
                nearby_epitopes = list(nearby_epitopes_set)
                # Sorting as sets are randomized even with random seed:
                nearby_epitopes.sort()
                if len(nearby_epitopes) > 0:
                    # Select any one epitope within distance:
                    nearby_epitope_idx = np.random.randint(0, len(nearby_epitopes))
                    nearby_epitope = nearby_epitopes[nearby_epitope_idx]
                    # Add epitope to train set:
                    train_selected_epitopes.append(nearby_epitope)
                    # Find index of train epitope in list of epitopes and delete to avoid sampling it in test set
                    nearby_epitope_real_idx = np.where(epitopes == nearby_epitope)[0]
                    epitopes = np.delete(epitopes, nearby_epitope_real_idx)
            else:
                # Discard the epitope:
                discarded_epitopes.append(curr_epitope)
                # Remove selected epitope from possible list:
                epitopes = np.delete(epitopes, curr_epitope_idx)
        if len(split_selected_epitopes) + len(discarded_epitopes) + len(
            train_selected_epitopes
        ) == len(epitopes_count_dict.keys()):
            raise Exception("No more values to sample from.")
        if epitopes.size == 0:
            if curr_count > (min_test_count * 0.5):
                continue
            else:
                raise Exception(
                    "Reached end of epitopes. No more values to sample from - relax your boundaries."
                )
    split_selected_epitopes = set(split_selected_epitopes)
    train_selected_epitopes = set(train_selected_epitopes)
    assert (
        len(split_selected_epitopes.intersection(train_selected_epitopes)) == 0
    ), "There is overlap between train and test. This should not happen."

    return split_selected_epitopes, train_selected_epitopes, curr_split_len


def split_by_hard(
    epitopes_count_dict: t.Dict, split_ratio: float = 0.8, lowest_percentile: float = 75
) -> (t.Set, t.Set, t.Set):
    """
    Implement hard split using the peptides only. If a peptide ends up in the test set,
    the peptide (and all its pairs) is removed from training set.

    Parameters
    ----------
    epitopes_count_dict: dict
        {epitope_seq: count}
    split_ratio: float
        Cutoff value for splitting the dataset

    Returns
    -------
    train_peptides_set: set
        Train peptides sequences
    test_peptides_set: set
        Test peptides sequences
    val_peptides_set: set
        Val peptides sequences
    """
    # Load dictionary to numpy array:
    epitopes_count_arr = np.array(list(epitopes_count_dict.items()))
    # Find inter-quartile range that contains the highest represented % of the data:
    counts = np.array(epitopes_count_arr[:, 1], dtype=int)
    high, low = np.percentile(counts, [100, lowest_percentile])
    print(f"Found values high {high} and low {low} for dataset.")
    # Select epitopes within the quartile range:
    epitopes_within_range = epitopes_count_arr[(counts <= high) & (counts >= low)]
    # Calculate test budget by summing all and multiplying by split ratio:
    min_test_val_count = int(counts.sum() * (1 - split_ratio)) // 2
    # Select the n shuffled peptides for test
    test_peptides_set, test_len = _fill_set(
        min_test_count=min_test_val_count,
        epitopes=epitopes_within_range[:, 0],  # Select only epitopes
        epitopes_count_dict=epitopes_count_dict,
    )
    # Calculate all peptides remaining after sampling:
    remaining_peptides_within_range_set = (
        set(epitopes_within_range[:, 0]) - test_peptides_set
    )
    # Select the n shuffled peptides for validation
    val_peptides_set, val_len = _fill_set(
        min_test_count=min_test_val_count,
        epitopes=np.array(
            list(remaining_peptides_within_range_set)
        ),  # Select only unused epitopes
        epitopes_count_dict=epitopes_count_dict,
    )
    all_peptides_set = set(epitopes_count_arr[:, 0])
    # Select train peptides that are not in test/val:
    train_peptides_set = all_peptides_set - test_peptides_set
    train_peptides_set -= val_peptides_set
    # Print metrics:
    print(f"Test contains {test_len} pairs  {(test_len/counts.sum()):.2%}")
    print(f"Validation contains {val_len} pairs  {(val_len/counts.sum()):.2%}")
    print(
        f"Training split contains: {len(train_peptides_set)} unique epitopes {(len(train_peptides_set)/len(all_peptides_set)):.2%}\n"
        f"Test split contains: {len(test_peptides_set)}  unique epitopes {(len(test_peptides_set)/len(all_peptides_set)):.2%}\n"
        f"Validation split contains: {len(val_peptides_set)}  unique epitopes {(len(val_peptides_set)/len(all_peptides_set)):.2%}"
    )

    return train_peptides_set, test_peptides_set, val_peptides_set


def split_by_difference(
    dataset: np.ndarray,
    epitopes_count_dict: dict,
    split_ratio: float = 0.8,
    highest_percentile: float = 100.0,
    lowest_percentile: float = 75.0,
) -> (t.Set, t.Set, t.Set):
    """
    Ensures that in the train and test set there's at least 100-lowest_percentile similarity

    Parameters
    ----------
    dataset: np.ndarray
        (seq1, seq2, metric: float) np.ndarray
    epitopes_count_dict: dict
        {epitope_seq: count}
    split_ratio: float
        Cutoff value for splitting the dataset
    lowest_percentile: float
        Lowest percentile of similarity. Default: 75 (ie. top 25%)

    Returns
    -------
    train_peptides_set: set
        Train peptides sequences
    test_peptides_set: set
        Test peptides sequences
    val_peptides_set: set
        Val peptides sequences

    """
    # Load dictionary to numpy array:
    epitopes_count_arr = np.array(list(epitopes_count_dict.items()))
    # Find inter-quartile range that contains the highest represented % of the data:
    counts = np.array(epitopes_count_arr[:, 1], dtype=int)
    # Find quartile containing the cutoff
    high, low = np.percentile(
        dataset.values[:, -1], [highest_percentile, lowest_percentile]
    )  # FIXME highest percentile is not always 100
    print(f"Found values high {high} and low {low} for dataset.")
    # Now all the candidates are within a distance:
    epitopes_within_range = dataset[
        (dataset.values[:, -1] <= high) & (dataset.values[:, -1] >= low)
    ]
    # Calculate test budget by summing all and multiplying by cutoff:
    min_test_val_count = int(counts.sum() * (1 - split_ratio)) // 2
    epitopes = np.unique(epitopes_within_range.values[:, 0])
    pairs_count = 0
    for e in epitopes:
        pairs_count += epitopes_count_dict[e]
    print(
        f"Found {len(epitopes)} to be in the between high {highest_percentile} and low {lowest_percentile} percentile. Unique epitopes and {pairs_count} Pairs {pairs_count/counts.sum():.2%} of all pairs"
    )
    # Select the n shuffled peptides for test
    test_peptides_set, temp_train_peptide_set, test_len = _fill_set_similarity(
        min_test_count=min_test_val_count,
        epitopes=epitopes,
        epitopes_count_dict=epitopes_count_dict,
        epitopes_within_range_df=epitopes_within_range,
    )
    # Calculate all peptides remaining after sampling:
    remaining_peptides_within_range_set = (
        set(epitopes_within_range.values[:, 0]) - test_peptides_set
    )
    remaining_peptides_within_range_set -= temp_train_peptide_set
    # Select the n shuffled peptides for validation
    val_peptides_set, temp_train_peptide_set2, val_len = _fill_set_similarity(
        min_test_count=min_test_val_count,
        epitopes=np.array(
            list(remaining_peptides_within_range_set)
        ),  # Select only unused epitopes
        epitopes_count_dict=epitopes_count_dict,
        epitopes_within_range_df=epitopes_within_range,
    )
    all_peptides_set = set(epitopes_count_arr[:, 0])
    # Select train peptides that are not in test/val:
    train_peptides_set = all_peptides_set - test_peptides_set
    train_peptides_set -= val_peptides_set
    # Print metrics:
    print(f"Test contains {test_len} pairs  {(test_len/counts.sum()):.2%}")
    print(f"Validation contains {val_len} pairs  {(val_len/counts.sum()):.2%}")
    print(
        f"Training split contains: {len(train_peptides_set)} unique epitopes {(len(train_peptides_set)/len(all_peptides_set)):.2%}\n"
        f"Test split contains: {len(test_peptides_set)}  unique epitopes {(len(test_peptides_set)/len(all_peptides_set)):.2%}\n"
        f"Validation split contains: {len(val_peptides_set)}  unique epitopes {(len(val_peptides_set)/len(all_peptides_set)):.2%}"
    )

    return train_peptides_set, test_peptides_set, val_peptides_set
