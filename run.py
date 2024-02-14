"""
File: figure_creation.py
Author: John E. Parker
Date: 13 February 2024

Description: Filters data and generates figures and corresponding statistics.
"""

import numpy as np
import pandas as pd
import sys
from scipy import stats
import argparse

# import user modules
from helpers import *
import figure_1_naive_d1
import figure_2_naive_gpe
import figure_3_dd_d1
import figure_5_dd_gpe
import figure_7_comparisons
import run_statistics

# Replace with absolute path to where STReaC toolbox repository resides on your machine
# sys.path.append("/path/to/streac")
sys.path.append("/Users/johnparker/streac")


# Filter d1 and gpe processed data by baseline z-score
def get_data(d1_path, gpe_path, zscore=3):
    """
    Splits data set into correspnoding naive and DD groups, then keeps baseline FR and CV
    cells with z-scores within zscore value.
    Returns d1_naive, d1_dd, gpe_naive, gpe_dd as dataframes

    d1_path\t:\tpath to d1 processed data set
    gpe_path\t:\tpath to gpe processed data set
    zscore\t:\tz-score to filter data by (default = 3)
    """

    # Read in gpe data from CSV
    csv = f"{gpe_path}/all_data.csv"
    df = pd.read_csv(csv)
    gpe_pv_cont_soma = df[(df["group"].str.contains("PV"))]

    # Read in d1 data from DSV
    csv = f"{d1_path}/all_data.csv"
    df = pd.read_csv(csv)
    d1_pv_cont_soma = df[(df["group"].str.contains("MSNs"))]
    d1_pv_cont_soma["mouse"] = d1_pv_cont_soma["mouse"].replace(
        ["6-OHDA", "Naive"], ["6-OHDA mice", "Naive mice"]
    )

    # Combine into one dat set
    all = pd.concat([gpe_pv_cont_soma, d1_pv_cont_soma])

    # Clean data (See methods)
    gpe_pv_cont_soma.loc[
        gpe_pv_cont_soma["neural_response"] == "biphasic IE", "neural_response"
    ] = "adapting inhibition"
    gpe_pv_cont_soma.loc[
        gpe_pv_cont_soma["neural_response"] == "biphasic EI", "neural_response"
    ] = "excitation"

    gpe_pv_cont_soma.loc[gpe_pv_cont_soma["Mouse#"] == "NN", "Mouse#"] = 0
    gpe_pv_cont_soma.loc[gpe_pv_cont_soma["Mouse#"] == "NN", "Mouse#"] = 0

    d1_pv_cont_soma.loc[
        d1_pv_cont_soma["neural_response"] == "biphasic IE", "neural_response"
    ] = "adapting inhibition"
    d1_pv_cont_soma.loc[
        d1_pv_cont_soma["neural_response"] == "biphasic EI", "neural_response"
    ] = "excitation"

    all.loc[all["neural_response"] == "biphasic IE", "neural_response"] = (
        "adapting inhibition"
    )
    all.loc[all["neural_response"] == "biphasic EI", "neural_response"] = "excitation"

    # Combine naive and dd data sets
    naive = pd.concat(
        [
            gpe_pv_cont_soma[gpe_pv_cont_soma["mouse"] == "Naive mice"],
            d1_pv_cont_soma[d1_pv_cont_soma["mouse"] == "Naive mice"],
        ]
    )
    dd = pd.concat(
        [
            gpe_pv_cont_soma[gpe_pv_cont_soma["mouse"] == "6-OHDA mice"],
            d1_pv_cont_soma[d1_pv_cont_soma["mouse"] == "6-OHDA mice"],
        ]
    )

    # Find naive and dd z-scores for baseline firing rate and CV
    dd["z_base_freq"] = stats.zscore(dd["pre_exp_freq"])
    naive["z_base_freq"] = stats.zscore(naive["pre_exp_freq"])
    dd["z_base_cv"] = stats.zscore(dd["pre_exp_cv"])
    naive["z_base_cv"] = stats.zscore(naive["pre_exp_cv"])

    # Fiter data based on z-score
    filter_dd = dd[
        (np.abs(dd["z_base_freq"]) <= zscore) & (np.abs(dd["z_base_cv"]) <= zscore)
    ]
    filter_naive = naive[
        (np.abs(naive["z_base_freq"]) <= zscore)
        & (np.abs(naive["z_base_cv"]) <= zscore)
    ]

    # Creat new dataframes based on filtered data
    d1_naive = filter_naive[(filter_naive["group"].str.contains("MSNs"))]
    d1_dd = filter_dd[(filter_dd["group"].str.contains("MSNs"))]
    gpe_naive = filter_naive[(filter_naive["group"].str.contains("PV"))]
    gpe_dd = filter_dd[(filter_dd["group"].str.contains("PV"))]

    return d1_naive, d1_dd, gpe_naive, gpe_dd


# directory to store produced figures
save_dir = "figures"
run_cmd(f"mkdir -p {save_dir}")

# Source of processed data
d1_processed_data = "./data/d1_msns/processed_data"
gpe_processed_data = "./data/gpe_pv/processed_data"

# Gather processed data after filtering by baseline z-score
(
    d1_naive,
    d1_dd,
    gpe_naive,
    gpe_dd,
) = get_data(d1_processed_data, gpe_processed_data, zscore=3)

# Allow for command line input for figure generation and stats results
parser = argparse.ArgumentParser(description="Reproduce figures and stats.")
parser.add_argument(
    "--figures",
    type=int,
    nargs="+",
    help="List of figure numbers to generate",
    default=[],
)
parser.add_argument(
    "--stats", help="output stats to statistics.txt", action="store_true"
)
args = parser.parse_args()


# If true, generate stats reported in paper
if args.stats:
    run_statistics.print_statistics(d1_naive, gpe_naive, d1_dd, gpe_dd, decimals=4)

# Generate figures if fig number is in array figures
if 1 in args.figures:
    figure_1_naive_d1.generate_figure(d1_naive, save_dir=save_dir)
if 2 in args.figures:
    figure_2_naive_gpe.generate_figure(gpe_naive, d1_naive, save_dir=save_dir)
if 3 in args.figures:
    figure_3_dd_d1.generate_figure(d1_dd, d1_naive, save_dir=save_dir)
if 5 in args.figures:
    figure_5_dd_gpe.generate_figure(gpe_dd, gpe_naive, save_dir=save_dir)
if 7 in args.figures:
    figure_7_comparisons.generate_figure(
        d1_naive,
        d1_dd,
        gpe_naive,
        gpe_dd,
        save_dir=save_dir,
    )
