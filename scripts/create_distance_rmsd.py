import argparse
from itertools import combinations, repeat
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_prep import (
    calculate_diff_between_seq,
    calculate_seq_metrics,
    clean_netsolp_outfile,
    run_netsolp_on_fasta,
    save_dict_to_csv,
    save_dict_to_fasta,
    save_dict_to_json,
    load_data,
)

import argparse
import tempfile
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import ampal
import numpy as np
import pymol
import argparse
import tempfile
from pathlib import Path

import ampal
import numpy as np
import pymol
from sklearn import metrics
from tqdm import tqdm


def calculate_RMSD_and_gdt(seq1_seq2, output_path) -> (float, float):
    # Code adapted from https://github.com/wells-wood-research/timed-design/tree/main/scripts
    seq1, seq2 = seq1_seq2
    pdb_original_path = output_path / (seq1 + ".pdb")
    pdb_predicted_path = output_path / (seq2 + ".pdb")
    pymol.pymol_argv = ["pymol", "-qc"]
    pymol.finish_launching()
    cmd = pymol.cmd
    cmd.undo_disable()  # Avoids pymol giving errors with memory
    cmd.delete("all")
    cmd.load(pdb_original_path, object="refori")
    cmd.load(pdb_predicted_path, object="modelled")
    sel_ref, sel_model = cmd.get_object_list("all")
    # Select only C alphas
    sel_ref += " and name CA"
    sel_model += " and name CA"
    rmsd = cmd.super(target=sel_ref, mobile=sel_model)[0]
    cmd.super(target=sel_ref, mobile=sel_model, cycles=0, transform=0, object="aln")
    mapping = cmd.get_raw_alignment("aln")
    cutoffs = [1.0, 2.0, 4.0, 8.0]
    distances = []
    for mapping_ in mapping:
        try:
            atom1 = f"{mapping_[0][0]} and id {mapping_[0][1]}"
            atom2 = f"{mapping_[1][0]} and id {mapping_[1][1]}"
            dist = cmd.get_distance(atom1, atom2)
            cmd.alter(atom1, f"b = {dist:.4f}")
            distances.append(dist)
        except:
            continue
    distances = np.asarray(distances)
    gdts = []
    for cutoff in cutoffs:
        gdt_cutoff = (distances <= cutoff).sum() / (len(distances))
        gdts.append(gdt_cutoff)

    mean_gdt = np.mean(gdts)
    return seq1, seq2, rmsd, mean_gdt


def main(args):
    args.input_path = Path(args.input_path)
    args.structure_path = Path(args.structure_path)
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    assert (
        args.structure_path.exists()
    ), f"Structure path {args.structure_path} does not exist"
    # Select correct database:
    _, epitopes_count_dict = load_data(args.input_path)
    unique_epitopes = set(list(epitopes_count_dict.keys()))
    # Extract sequence metrics in multiprocessing loop:
    with Pool(processes=args.workers) as p:
        results_list = p.map(calculate_seq_metrics, unique_epitopes)
        p.close()
    # Flatten dictionary:
    res_dict = {}
    for curr_res in results_list:
        curr_dict, _ = curr_res
        if curr_dict is not None:
            res_dict.update(curr_dict)
    # Save data:
    output_file = args.input_path.parent / f"{args.input_path.stem}_epitopes.fasta"
    save_dict_to_fasta(res_dict, output_file)
    csv_columns = "epitope, length, charge, iso_ph, molecular_weight, extinction_coeff"
    # Save to json and csv
    output_csv = args.input_path.parent / f"{args.input_path.stem}_epitopes.csv"
    save_dict_to_csv(res_dict=res_dict, outpath=output_csv, columns=csv_columns)
    output_json = args.input_path.parent / f"{args.input_path.stem}_epitopes.json"
    save_dict_to_json(res_dict=res_dict, outpath=output_json)
    # Create distances:
    seqs = list(res_dict.keys())
    # Make all cominations of epitopes sequences
    seq_combo = list(combinations(seqs, 2))
    print(len(seq_combo))
    with Pool(processes=args.workers) as p:
        distance_matrix = p.starmap(
            calculate_RMSD_and_gdt,
            zip(seq_combo, repeat(args.structure_path)),
        )
        p.close()
    # Non multiprocessing:
    # distance_matrix = []
    # for seq in seq_combo:
    #     c = calculate_diff_between_seq(res_dict, seq)
    #     distance_matrix.append(c)
    # Convert to Numpy array and remove nan
    distance_matrix = pd.DataFrame(distance_matrix)
    print(distance_matrix.shape)
    distance_matrix.dropna(inplace=True)
    print(distance_matrix.shape)
    distance_output = args.input_path.parent / f"{args.input_path.stem}_distance_rmsd.csv"
    # Add alignment columns
    new_columns_list = csv_columns.split(",")[1:]
    new_columns = (
        "seq1, seq2, RMSD, GDT"
    )
    np.savetxt(
        distance_output,
        distance_matrix.values,
        delimiter=",",
        fmt="%s",
        header=new_columns,
        comments="",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to input file")
    parser.add_argument(
        "--structure_path", type=str, help="Path to pdb structures files"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers to use (default: 8)"
    )
    params = parser.parse_args()
    main(params)
