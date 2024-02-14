"""
File: run_classification.py
Author: John E. Parker
Date: 14 February 2024

Description: Runs STReaC toolbox on data.
"""

from helpers import *

# change this path to the approriate string
data_direc = "/path/to/Aristieta_Parker_Gao_Gittis_Rubin_2024_D1_GPe/data"

# Designate which data sets to run (both true for figure reproduction)
d1 = True
gpe = True

# Parameters used in data analysis for paper
baseline = [1, 1]
stim = [0, 1]
binwidth = 0.05

# Run analysis
if d1:
    run_cmd(
        f"python analyze_data.py -d {data_direc}/d1_msns/pre_processed_data -r {data_direc}/d1_msns/processed_data -b {baseline[0]} {baseline[1]} -l {stim[0]} {stim[1]} -g -ar -pd -bw {binwidth}"
    )

if gpe:
    run_cmd(
        f"python analyze_data.py -d {data_direc}/gpe_pv/pre_processed_data -r {data_direc}/gpe_pv/processed_data -b {baseline[0]} {baseline[1]} -l {stim[0]} {stim[1]} -g -ar -pd -bw {binwidth}"
    )
