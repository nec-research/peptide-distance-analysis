import argparse
import json
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.Align import substitution_matrices
from ampal.amino_acids import standard_amino_acids
from ampal.analyse_protein import (
    sequence_charge,
    sequence_isoelectric_point,
    sequence_molar_extinction_280,
    sequence_molecular_weight,
)

from netsolp.predict import get_preds_distilled


def load_data(input_path):
    if "vdjdb" in str(input_path):
        df = pd.read_csv(input_path, skipinitialspace=True)
        # filtering viruses for humans
        df = df[df["mhc.class"] == "MHCI"]
        df = df[df["species"] == "HomoSapiens"]
        df = df[df["vdjdb.score"] >= 0]
        df = df[
            df["antigen.species"].isin(
                ["SARS-CoV-2", "InfluenzaA", "HIV-1", "HIV", "HCV", "CMV"]
            )
        ]
        df = df[df["cdr3.beta"].notna()]
        df = df[df["antigen.epitope"].notna()]
        # Extract epitopes
        epitopes = df["antigen.epitope"].values
    elif "McPAS-TCR" in str(input_path):
        df = pd.read_csv(input_path, encoding="unicode_escape", skipinitialspace=True)
        raise NotImplementedError
    elif "iedb" in str(input_path):
        df = pd.read_csv(input_path, skipinitialspace=True, skiprows=1)
        # Extract epitopes
        epitopes = df["Description"].values
    else:
        raise NotImplementedError
    # Count Epitopes:
    unique, counts = np.unique(epitopes, return_counts=True)
    epitopes_count_dict = dict(zip(unique, counts))
    return df, epitopes_count_dict


def sanitize_seq(seq: str) -> str:
    """
    Deals with non-standard characters in the amino acid sequence.

    eg. "RYGFVANF + OX(F4)" or "RYGFVANB"

    Parameters
    ----------
    seq: str
        Amino acid sequence

    Returns
    -------
    seq: str
        Cleaned  amino acid sequence

    """
    # Deals with "RYGFVANF + OX(F4)" -> "RYGFVANF"
    seq = seq.split(" ")[0]
    # TODO: We could also add this to remove non-(capital)-letters but the current strategy works for iedb
    # seq = re.sub('[^A-Z]+', '', seq)
    # Deals with non-standard amino acids:
    # seq -> s, e, q
    used_residues = set(list(seq))
    residues = set(list(standard_amino_acids.keys()))
    # If used - residues is longer than the sequence contains non-standard amino acids
    diff_residues = used_residues - residues
    # If non-standard
    if len(diff_residues) > 0:
        return False
    else:
        return seq


def calculate_seq_metrics(seq: str) -> t.Tuple[t.Dict, t.List]:
    """
    Calculates sequence metrics.

    Currently only supports: Charge at pH 7, Isoelectric Point, Molecular Weight
    Own Code From: https://github.com/wells-wood-research/timed-design/blob/4cf4812b7f3b675747c16216a7f5c66c0f22ed05/design_utils/analyse_utils.py
    Parameters
    ----------
    seq: str
        Sequence of residues

    Returns
    -------
    metrics: t.Tuple[t.Dict, t.List]
        (res_dict, [seq, charge , iso_ph, mw, me])
    """
    seq = sanitize_seq(seq)
    if not seq:
        res_dict = {seq: (np.nan, np.nan, np.nan, np.nan, np.nan)}
        return res_dict, list(res_dict.values())
    else:
        charge = sequence_charge(seq)
        iso_ph = sequence_isoelectric_point(seq)
        mw = sequence_molecular_weight(seq)
        me = sequence_molar_extinction_280(seq)
        res_dict = {seq: (len(seq), charge, iso_ph, mw, me)}
        return res_dict, list(res_dict.values())


def save_dict_to_json(res_dict: t.Dict, outpath: Path):
    """
    Save dictionary to JSON

    Parameters
    ----------
    res_dict: t.Dict
        Dictionary of results
    outpath: Path
        Path to save file
    """
    # Ensure that output path is csv:
    if outpath.suffix == ".json":
        pass
    else:
        outpath.suffix = ".json"
    with open(outpath, "w") as outfile:
        json.dump(res_dict, outfile)


def save_dict_to_csv(res_dict: t.Dict, outpath: Path, columns: t.Optional[str] = None):
    """
    Save dictionary to csv

    Parameters
    ----------
    res_dict: t.Dict
        Dictionary of results
    outpath: Path
        Path to save file
    columns: t.Optional[str]
        Columns to save in the csv file
    """
    # Ensure that output path is csv:
    if outpath.suffix == ".csv":
        pass
    else:
        outpath.suffix = ".csv"
    # Add return in case it is missing
    if columns[-2:] == "\n":
        pass
    else:
        columns += "\n"
    # Save to CSV
    with open(outpath, "w") as outfile:
        outfile.write(columns)
        for key, values in res_dict.items():
            vals_str = ",".join(map(str, values))
            outfile.write(f"{key},{vals_str}\n")
    print(f"Saved csv to {outpath}")


def save_dict_to_fasta(
    res_dict: t.Dict,
    outpath: Path,
):
    """
    Save dictionary to fasta

    Parameters
    ----------
    res_dict: t.Dict
        Dictionary of results
    outpath: Path
        Path to save file
    """
    # Ensure that output path is csv:
    if outpath.suffix == ".fasta":
        pass
    else:
        outpath.suffix = ".fasta"
    with open(outpath, "w") as f:
        for epi, res in res_dict.items():
            f.write(f">{epi},{*res,}\n{epi}\n")
    print(f"Saved fasta to {outpath}")


def run_netsolp_on_fasta(
    fasta_path: str, models_path: str, output_path: str, workers: int = 8
):
    """
    Run netsolp on fasta file. Generates an argparse call so that everything runs at once.

    Parameters
    ----------
    fasta_path: str
        Path to Fasta to analyse
    models_path: str
        Path to netsolp model *folder*
    output_path: str
        Path to save netsolp results
    workers: int
        Number of cores to use

    """
    arguments = [
        "--FASTA_PATH",
        fasta_path,
        "--MODELS_PATH",
        models_path,
        "--PREDICTION_TYPE",
        "SU",
        "--OUTPUT_PATH",
        output_path,
        "--NUM_THREADS",
        str(workers),
    ]
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--FASTA_PATH")
    parser.add_argument("--MODELS_PATH")
    parser.add_argument("--PREDICTION_TYPE")
    parser.add_argument("--OUTPUT_PATH")
    parser.add_argument("--NUM_THREADS")
    arg = parser.parse_args(args=arguments)
    arg.NUM_THREADS = int(arg.NUM_THREADS)
    get_preds_distilled(arg)
    print("Finished running Netsolp")


def clean_netsolp_outfile(input_path: Path) -> dict:
    """
    Clean netsolp outfile from unwanted characters and merge into a larger dict

    Parameters
    ----------
    input_path: Path
        Path to netsolp analysis

    Returns
    -------
    res_dict: dict
        Results dictionary with netsolp + sequence metrics

    """
    res_dict = {}
    with open(input_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            # Skip Header
            if i == 0:
                continue
            else:
                # Remove useless characters
                result = (
                    line.replace('"', "")
                    .replace("\n", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(">", "")
                )
                results = result.split(",")
                # Create empty list in dict
                seq = results[0]
                res_dict[seq] = []
                for res in results[1:]:
                    if res != seq:
                        # Only append floating number and avoid appending the sequence twice
                        res_dict[seq].append(float(res))
    return res_dict


def align_seq(seq1: str, seq2: str) -> t.List[float]:
    """
    Align two sequences of any length using biopython pairwise align.

    Four alignments are made:
        - localxx
        - localds
        - globalxx
        - globalds
    See: https://biopython.org/docs/1.76/api/Bio.pairwise2.html?highlight=localds for details

    Parameters
    ----------
    seq1: str
        Peptide sequence
    seq2: str
        Peptide sequence

    Returns
    -------
    scores: t.List[float]
        Scores in the order [glob_align_ds, glob_align_xx, loc_align_ds, loc_align_xx]
    """
    blosum62 = substitution_matrices.load("BLOSUM62")
    # Local alignments
    # loc_align_xx = pairwise2.align.localxx(seq2, seq1)
    loc_align_xx = pairwise2.align.localxx(seq2, seq1)
    # Values taken from https://biopython.org/docs/1.76/api/Bio.pairwise2.html?highlight=localds
    loc_align_ds = pairwise2.align.localds(seq2, seq1, blosum62, -10, -1)
    # Global alignments
    glob_align_xx = pairwise2.align.globalxx(seq2, seq1)
    glob_align_ds = pairwise2.align.globalds(seq2, seq1, blosum62, -10, -1)
    # Sometimes alignment may fail for weird reasons
    scores = []
    for aln in [glob_align_ds, glob_align_xx, loc_align_ds, loc_align_xx]:
        if len(aln) > 0:
            scores.append(aln[0].score)
        else:
            scores.append(np.nan)

    return scores


def calculate_diff_between_seq(
    res_dict: t.Dict, seq1_seq2_tuple: t.Tuple[str, str]
) -> t.List[t.Union[str, str, t.Optional[float]]]:
    """
    Calculates difference in values between seq metrics of 2 peptides.
    It also creates a list with seq1, seq2, and various differences + alignment scores

    Parameters
    ----------
    res_dict: t.Dict
        Dictionary of results
    seq1_seq2_tuple: t.Tuple[str, str]
        Tuple of two peptide sequences

    Returns
    -------
    result: t.List[str, str, t.Optional[int, float]]
        Results seq1, seq2, *mse_feats, *aln_scores

    """
    seq1, seq2 = seq1_seq2_tuple
    # Align sequences
    aln_scores = align_seq(seq1, seq2)
    # Calculate difference between sequence features:
    seq1_feats = np.array(res_dict[seq1])
    seq2_feats = np.array(res_dict[seq2])
    mse_feats = np.square(seq1_feats - seq2_feats)
    return [seq1, seq2, *mse_feats, *aln_scores]
