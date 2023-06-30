import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")
sns.color_palette("Set2")


def plot_histograms(curr_results: pd.DataFrame, title: str):
    """
    Plot histograms of floating values of results
    Parameters
    ----------
    curr_results: pd.DataFrame
        Results in dataframe (needs numerical values)
    title: str
        Title to add to the plot
    """
    # Get all columns with numerical values:
    numerical_cols = list(curr_results.select_dtypes(include=[float]).columns.values)
    fig, axs = plt.subplots(
        ncols=len(numerical_cols), figsize=(8 * len(numerical_cols), 8)
    )
    for i, num_col in enumerate(numerical_cols):
        ax = sns.histplot(curr_results, x=num_col, ax=axs[i])
        ax.set_title(f"{num_col} Difference")
        # Fix limits to make plots nicer.
        # TODO: Improve?
        if num_col == " solubility":
            ax.set_xlim(-0.0001, 0.025)

        if num_col == " charge":
            ax.set_xlim(-0.0001, 40)

        if num_col == " molecular_weight":
            ax.set_xlim(-0.0001, 0.1 * 1e6)

        if num_col == " usability":
            ax.set_xlim(-0.0001, 0.0045)

        if num_col == " localxx":
            ax.set_xlim(-32, 20)

        if num_col == " globalxx":
            ax.set_xlim(0, 25)

        if num_col == " globalds":
            ax.set_xlim(0, 8)

    plt.tight_layout()
    plt.title(title)
    plt.savefig(f"{title}_histograms_float.png")
    plt.close()


def plot_countplot(curr_results: pd.DataFrame, title: str):
    """
    Plot countplot for integer values. Values are ordered by count

    Parameters
    ----------
    curr_results: pd.DataFrame
        Results in dataframe (needs numerical values)
    title: str
        Title to add to the plot
    """
    # Get all columns with numerical values:
    numerical_cols = list(curr_results.select_dtypes(include=[int]).columns.values)
    fig, axs = plt.subplots(
        ncols=len(numerical_cols), figsize=(8 * len(numerical_cols), 8)
    )
    for i, num_col in enumerate(numerical_cols):
        # Fix font for localxx
        if num_col == " localxx":
            length_col = curr_results[num_col].value_counts().iloc[:50].index
            font = 7
        else:
            length_col = curr_results[num_col].value_counts().index
            font = 10
        ax = sns.countplot(x=curr_results[num_col], order=length_col, ax=axs[i])
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=90, ha="center", fontsize=font
        )
        ax.set_title(f"{num_col} Difference")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title}_histograms_int.png")
    plt.close()


def plot_pairplot(curr_results: pd.DataFrame, title: str):
    """
    Plot a pairplot for the whole dataset.

    Parameters
    ----------
    curr_results: pd.DataFrame
        Results in dataframe (needs numerical values)
    title: str
        Title to add to the plot
    """
    g = sns.pairplot(
        curr_results,
        markers="+",
        diag_kind="hist",
        kind="reg",
        plot_kws={"line_kws": {"linestyle": "--"}, "scatter_kws": {"alpha": 0.01}},
    )
    g = g.map_lower(sns.kdeplot, levels=4, color="0")
    g = g.fig.suptitle(title)
    plt.savefig(f"{title}_pairplot.png")
    plt.close()


def main(args):
    args.input_path = Path(args.input_path)
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    curr_results = pd.read_csv(args.input_path)
    plot_histograms(curr_results, args.plot_title)
    plot_countplot(curr_results, args.plot_title)
    plot_pairplot(curr_results, args.plot_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to input file")
    parser.add_argument("--plot_title", type=str, help="Plot Title")
    params = parser.parse_args()
    main(params)
