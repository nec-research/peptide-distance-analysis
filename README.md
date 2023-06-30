# Assessing the Generalization Capabilities of TCR Binding Predictors via Peptide Distance Analysis

## Citation

Coming soon

## Steps to Reproduce

### Raw Data

#### VDJDB

This repository requires you to download VDJdb in `.csv` format. We select the following settings:

```
A viral subset of the VDJdb dataset, focusing on human host and MHC class I. We omitted MHC class II due to low samples available. The dataset includes peptides from SARS-CoV-2, Influenza A, Human Immunodeficiency Virus (HIV), Hepatitis C Virus (HCV) and Cytomegalovirus (CMV). We discarded data points which do not include both the CDR3-B and the peptide.

The dataset contains 52 unique MHC A and 1 MHC B alleles, 16,504 unique CDR3-A sequences, 28,831 unique CDR3-B sequences and 757 unique peptides, for a total of 34,415 binding samples.
```

Note: this code should work irrespectively of the filter chosen.

#### 3D Structures of Epitopes (optional)

Testing the distance of the 3D structure between epitopes requires generating them through folding methods like AlphaFold, OmegaFold, or ESMFold. 

We used the ESMFold API to obtain the PDB files:

```
curl -X POST --data "INSERT_EPITOPE_SEQUENCE" https://api.esmatlas.com/foldSequence/v1/pdb/ > path/to/INSERT_EPITOPE_SEQUENCE.pdb
sleep 60
```

### Distance Data

#### Install the Repository

You can install the repository with the `setup.sh` file:

```
bash setup.sh
```

This should take care of all the dependencies and NetSolP. Alternatively use conda:

```
conda env create -f environment.yml
```


#### Sequence Distance

Use the file `scripts/create_distance_seq.py` to create sequence distances (LocalDS, GlobalDS, LocalXX and GlobalXX) . Use as such:


```
python create_data.py --input_path path/to/viral_vdjdb_full.csv --workers 16
```

The code also optionally supports NetSolP as a possible metric by providing the path as `--netsolp_model PATH`

#### Shape Distance

Use the file `scripts/create_distance_rmsd.py` to create RMSD shape distances. Use as such:


```
python create_data.py --input_path path/to/viral_vdjdb_full.csv  --structure_path path/to/structures_folder/ --workers 16
```

### Split Creation


At this point you should have downloaded VDJdb and have one or more distance files. We will use these to create multiple split ranges. In this example we selected the ranges 0-33, 33-66, 66-100. Place a bash script similar to this in your `scripts` folder

````
python create_multiple_splits.py --input_path path/to/distance_file.csv --dataset_path path/to/vdjdb.csv --metric_to_split your_metric --tt_fig_out tt_fig_yourmetric_0 --tv_fig_out tv_fig_yourmetric_0 --lowest_percentiles 0 33 66 --highest_percentiles 33 66 100 --seed 0 --split_ratio .90 --max_epitope_count 5000 --min_epitope_count 5 --leeway 0.01
python create_multiple_splits.py --input_path path/to/distance_file.csv --dataset_path path/to/vdjdb.csv --metric_to_split your_metric --tt_fig_out tt_fig_yourmetric_1 --tv_fig_out tv_fig_yourmetric_1 --lowest_percentiles 0 33 66 --highest_percentiles 33 66 100 --seed 1 --split_ratio .90  --max_epitope_count 5000 --min_epitope_count 5 --leeway 0.01
python create_multiple_splits.py --input_path path/to/distance_file.csv --dataset_path path/to/vdjdb.csv --metric_to_split your_metric --tt_fig_out tt_fig_yourmetric_2 --tv_fig_out tv_fig_yourmetric_2 --lowest_percentiles 0 33 66 --highest_percentiles 33 66 100 --seed 2 --split_ratio .90 --max_epitope_count 5000 --min_epitope_count 5 --leeway 0.01
python create_multiple_splits.py --input_path path/to/distance_file.csv --dataset_path path/to/vdjdb.csv --metric_to_split your_metric --tt_fig_out tt_fig_yourmetric_3 --tv_fig_out tv_fig_yourmetric_3 --lowest_percentiles 0 33 66 --highest_percentiles 33 66 100 --seed 3 --split_ratio .90 --max_epitope_count 5000 --min_epitope_count 5 --leeway 0.01
python create_multiple_splits.py --input_path path/to/distance_file.csv --dataset_path path/to/vdjdb.csv --metric_to_split your_metric --tt_fig_out tt_fig_yourmetric_4 --tv_fig_out tv_fig_yourmetric_4 --lowest_percentiles 0 33 66 --highest_percentiles 33 66 100 --seed 4 --split_ratio .90 --max_epitope_count 5000 --min_epitope_count 5 --leeway 0.01

```

A thorough description of all the values is available in the `create_multiple_splits.py` file.