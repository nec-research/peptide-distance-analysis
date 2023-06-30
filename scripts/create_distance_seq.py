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


def main(args):
    args.input_path = Path(args.input_path)
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
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
    # Run Netsolp
    if args.netsolp_model:
        output_csv = args.input_path.parent / f"{args.input_path.stem}_netsolp.csv"
        model_path = Path(args.netsolp_model)
        assert model_path.exists(), f"Model file {model_path} does not exist"
        run_netsolp_on_fasta(
            fasta_path=str(output_file),
            models_path=str(model_path),
            output_path=str(output_csv),
        )
        res_dict = clean_netsolp_outfile(output_csv)
        csv_columns += ", solubility, usability"
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
            calculate_diff_between_seq,
            zip(repeat(res_dict), seq_combo),
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
    distance_output = args.input_path.parent / f"{args.input_path.stem}_distance_seq.csv"
    # Add alignment columns
    new_columns_list = csv_columns.split(",")[1:]
    new_columns = (
        "seq1, seq2,"
        + ",".join(new_columns_list)
        + ", localxx, localds, globalxx, globalds"
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
        "--workers", type=int, default=8, help="Number of workers to use (default: 8)"
    )
    parser.add_argument(
        "--netsolp_model",
        type=str,
        nargs="?",
        help="Path to models to use Netsolp (default: None)",
    )
    params = parser.parse_args()
    main(params)
