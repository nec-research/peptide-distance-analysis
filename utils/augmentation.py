import typing as t
from itertools import repeat
from multiprocessing import Pool

import blosum
import numpy as np
import pandas as pd
from ampal.amino_acids import standard_amino_acids
from tqdm import tqdm

from utils.data_prep import load_data


def random_choice_prob_index(
    probs: np.ndarray,
    axis: int = 1,
    return_seq: bool = True,
) -> np.ndarray:
    """
    Samples from a probability distribution and returns a sequence or the indeces sampled.
    Code originally written in by me: https://github.com/wells-wood-research/timed-design/blob/4cf4812b7f3b675747c16216a7f5c66c0f22ed05/design_utils/sampling_utils.py#L53
    Code adapted from: https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a?noredirect=1&lq=1

    Parameters
    ----------
    probs: np.ndarray
        2D Array with shape (n, n_categries) where n is the number of residues.
    axis: int
        Axis along which to select.
    return_seq: bool
        Whether to return a residue sequence (True) or the index (False)


    Returns
    -------
    sequence: np.ndarray
        Sequence of residues or indeces sampled from a distribution
    """
    r = np.expand_dims(np.random.rand(probs.shape[1 - axis]), axis=axis)
    idxs = (probs.cumsum(axis=axis) > r).argmax(axis=axis)
    if return_seq:
        res = np.array(list(standard_amino_acids.keys()))
        return res[idxs]
    else:
        return idxs


def create_blosum_encoding(seq: str, blosum_threshold: int = 90) -> np.ndarray:
    """
    Encode sequence as probability distribution of mutation obtained from blosum

    Parameters
    ----------
    seq: str
        String of amino acids
    blosum_threshold: int
        Threshold for Blosum. Default: 90. n ϵ {45,50,62,80,90}

    Returns
    -------
    encoding: np.ndarray
        (n, 20) array of probabilities where n is the sequence of residues of the epitope

    """
    matrix = blosum.BLOSUM(blosum_threshold)
    encoding = []
    # For each residue in the epitopes
    for res in seq:
        curr_encoding = []
        # Check probability of mutating to any of the 20:
        for aa in standard_amino_acids.keys():
            couple = f"{res}{aa}"
            curr_encoding.append(matrix[couple])
        encoding.append(curr_encoding)
    # Convert to numpy:
    encoding = np.array(encoding)
    # Invert blosum: blosum score = 2log_2(value) https://en.wikipedia.org/wiki/BLOSUM
    encoding = np.exp(encoding / 2)
    # Divide by sum of row to get values between 0 - 1
    encoding = (encoding + np.abs(np.min(encoding))) / (
        np.max(encoding) + np.abs(np.min(encoding))
    )
    # Normalise along row so that rows sum up to 1
    sum_enc = np.sum(encoding, axis=1)
    encoding = encoding / sum_enc[:, None]

    return encoding


def sample_from_sequences(seq: str, sample_n: int) -> t.Dict:
    """
    Given a sequence of amino acid, encode it with blosum and sample from it.

    Parameters
    ----------
    seq: str
        Sequence of amino acid to augment
    sample_n: int
        Number of augmented sequences to make for the epitope

    Returns
    -------
    augmented_epitopes_dict: dict
        Dictionary of {sequence: [augmented_sequences]}
    """
    # Sample from distribution
    sampled_seq_set = set()
    encoded_seq = create_blosum_encoding(seq)
    # Create samples until we have sample n
    while len(sampled_seq_set) < sample_n:
        seq_list = random_choice_prob_index(encoded_seq, return_seq=True)
        # Join seq from residue list to one string
        sampled_seq = "".join(seq_list)
        if sampled_seq not in sampled_seq_set:
            sampled_seq_set.add(sampled_seq)
    augmented_epitopes_dict = {seq: list(sampled_seq_set)}
    return augmented_epitopes_dict


def augment_epitopes(
    epitopes: np.ndarray, augmentation_factor: t.Union[t.List, int], workers: int = 16
):
    """
    Augment array of epitopes by augmentation_factor using multiprocessing

    Parameters
    ----------
    epitopes: np.ndarray
        List of epitopes
    augmentation_factor: t.Union[t.List, int]
        Augmentation factor. If augmentation factor is a list, each epitope will be augmented by the corresponding count
    workers: int
        Number of processes to launch

    Returns
    -------
    epitope_to_augmented: dict
        Epitope to augmented epitopes
    """
    if isinstance(augmentation_factor, int):
        repeated_augmentation_factor = repeat(augmentation_factor)
    elif isinstance(augmentation_factor, list):
        assert len(augmentation_factor) == len(
            epitopes
        ), f"Augmentation factor was list, but got length {len(augmentation_factor)} but have epitopes of length {len(epitopes)}"
        repeated_augmentation_factor = augmentation_factor
    else:
        raise ValueError(f"Got augmentation_factor of type {type(augmentation_factor)}")

    with Pool(processes=workers) as p:
        epitope_to_metric_list = p.starmap(
            sample_from_sequences,
            zip(
                epitopes,
                repeated_augmentation_factor,
            ),
        )
        p.close()
    # Flatten dictionary:
    epitope_to_augmented = {}
    for curr_dict in epitope_to_metric_list:
        if curr_dict is not None:
            epitope_to_augmented.update(curr_dict)
    return epitope_to_augmented


def _merge_all_augmented_epitopes(copy_df, augmented_epitope):
    original_epitope = copy_df["antigen.epitope"].unique()[0]
    copy_df["antigen.epitope"] = augmented_epitope
    copy_df["antigen.original"] = original_epitope
    return copy_df


def augment_epitopes_df(
    df: pd.DataFrame,
    augmentation_factor: int,
    is_uniform: bool = False,
    workers: t.Optional[int] = None,
) -> pd.DataFrame:
    """
    Augment dataframe of epitopes by augmentation factor.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas dataframe compatible with vdjdb.
    augmentation_factor: int
        Number of times data is to be augmented.
        If 1, uniform augmentation is performed where the datapoints below the 80th percentile are increased evenly

    Returns
    -------
    augmented_df: pd.Dataframe
        Pandas dataframe with augmented epitopes
    """
    assert augmentation_factor > 1 or is_uniform, "Augmentation factor must be above 1."
    augmented_df = df.copy()
    epitopes = df["antigen.epitope"].values
    # Count Epitopes:
    unique_epitopes, counts = np.unique(epitopes, return_counts=True)
    if is_uniform:
        # Multiply by augmentation factor
        new_counts = counts * augmentation_factor
        # Calculate top 20 and make sure all counts below are set to top20
        top_count = np.percentile(new_counts, 80)
        new_counts[new_counts <= top_count] = top_count
        # Calculate difference of how many are needed per epitopes
        sample_n = new_counts - counts
        sample_n = list(sample_n)
    else:
        sample_n = augmentation_factor - 1

    epitope_to_augmented = augment_epitopes(unique_epitopes, sample_n)
    for e in tqdm(unique_epitopes, desc="Augmenting epitopes:"):
        # Copy dataframe for current epitope to save space
        copy_df = df[df["antigen.epitope"] == e].copy()
        # For each augmented epitope create new augmented dataframe and merge with original
        if workers:
            # Use multiprocessing:
            with Pool(processes=workers) as p:
                all_augmented_df = p.starmap(
                    _merge_all_augmented_epitopes,
                    zip(
                        repeat(copy_df.copy()),
                        epitope_to_augmented[e],
                    ),
                )
                p.close()
        else:
            # No workers, just loop
            all_augmented_df = []
            # For each augmented epitope
            for a_e in epitope_to_augmented[e]:
                # In a copy, replace old epitope with augmented epitope
                curr_df = _merge_all_augmented_epitopes(
                    copy_df.copy(),
                    a_e,
                )
                all_augmented_df.append(curr_df)
        # Merge all dataframes together
        augmented_df = pd.concat(
            [augmented_df] + all_augmented_df,
            ignore_index=True,
        )

    return augmented_df

# if __name__ == "__main__":
#     df, _ = load_data("/Users/leo/Desktop/viral_vdjdb_full2.csv")
#     a = augment_epitopes_df(df, 2, True)
#     print(a)
