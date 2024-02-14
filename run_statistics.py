"""
File: run_statistics.py
Author: John E. Parker
Date: 14 February 2024

Description: Performs statistical tests on data and prints out results.
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import analyze_data
import sys


def compare_responses(x1, x2):
    """
    Runs fisher exact test for distriubtion of responses. Returns list of p-vals.

    x1\t:\tfirst data set
    x2\t:\tsecond data set
    """

    # Calcuate 2 x 2 table, perform fisher exact for x1 and x2 for each category within x1
    pvals = [
        stats.fisher_exact(
            [[x1[i], np.sum(x1) - x1[i]], [x2[i], np.sum(x2) - x2[i]]]
        ).pvalue
        for i in range(len(x1))
    ]
    return pvals


def compare_response_dist(x1, x2):
    """
    Returns p-val for chi^2 between distriubtion of responses.

    x1\t:\tfirst data set
    x2\t:\tsecond data set
    """
    return stats.chi2_contingency(np.asarray([x1, x2])).pvalue

def compare_locs(x1):
    """
    Returns p-val for chi^2 location distributions of responses.

    x1\t:\tfirst data set with medial, central, and lateral response
    """
    return stats.chi2_contingency(x1).pvalue


def compare_locs_between(x1, x2):
    """
    Returns p-val for chi^2 between two locataion distribution of responses.

    x1\t:\tfirst data set
    x2\t:\tsecond data set
    """
    return stats.chi2_contingency(np.asarray([x1, x2])).pvalue


def pval_string(pval, paired=False):
    """
    Returns string signifiying level of signficance for p-val.

    pval\t:\tp-val to determine level of signficance
    paried\t:\tif true return '+' for signficance levels, otherwise '*' (default = False)
    """
    if pval < 0.001:
        return "+++" if paired else "***"
    elif pval < 0.01:
        return "++" if paired else "**"
    elif pval < 0.05:
        return "+" if paired else "*"
    else:
        return "n.s."

def compare_stat_prestim_stim(data, category1, category2, decimals=2,file="output.txt"):
    """
    Runs paired t-test on two quantites within data.

    data\t:\tdata set to run t-test on
    category1\t:\tfirst category to look at, usually pre-stim value
    category2\t:\tsecond category to look at, usually stim value
    deicmals\t:\tnumber of deicmals to round to (default = 2)
    file\t:\tfile to write results to
    """

    # Run paired t-test on categories
    _, pval = stats.ttest_rel(data[category1], data[category2])

    # Print results of p-val and mean +/- SD for each category
    print(f"{pval_string(pval,paired=True)}\tpval: {pval}\tPre-Stim mean: {np.round(np.mean(data[category1]),decimals)} +\\- {np.round(np.std(data[category1]),decimals)}\t Stim mean:{np.round(np.mean(data[category2]),decimals)} +\\- {np.round(np.std(data[category2]),decimals)}",file=file)


def compare_dists_ttest(data1, data2):
    """
    Return p-val for unpaired t-test between two distributions

    data1\t:\tfirst distribution of data
    data2\t:\tsecond distribution of data
    """

    # Remove NANs
    x1 = np.asarray(data1)
    x2 = np.asarray(data2)
    x1 = x1[np.logical_not(np.isnan(x1))]
    x2 = x2[np.logical_not(np.isnan(x2))]

    # Run and return p-val
    _, pval = stats.ttest_ind(
        x1, x2, alternative="less" if np.mean(x1) < np.mean(x2) else "greater"
    )
    return pval


def print_statistics(
    d1_naive_short, gpe_naive_short, d1_dd_short, gpe_dd_short, decimals=4
):
    """
    Reads in all data sets and run statstics. Writes results to file.
    """

    with open("statistics.txt", "w") as f:
        print("######### Figure 1 Stats #########",file=f)
        print("D1-ctl pre-stim FR vs stim FR:",file=f)
        _, d1_naive_norm_frs = analyze_data.get_norm_frs(d1_naive_short)
        compare_stat_prestim_stim(
            d1_naive_short, "baseline_freq", "stim_freq", decimals=decimals,file=f
        )


        print("\nD1-ctl pre-stim CV vs stim CV:",file=f)
        compare_stat_prestim_stim(
            d1_naive_short[d1_naive_short["neural_response"] != "complete inhibition"],
            "baseline_cv",
            "stim_cv",
            decimals=decimals,file=f
        )

        d1_naive_responses = analyze_data.get_response_distribution(d1_naive_short)
        d1_naive_base_FR_mfs, d1_naive_inh_mfs, d1_naive_ex_mfs, d1_naive_ne_mfs = (
            analyze_data.get_modulation_factors(d1_naive_short,"frequency")
        )

        d1_naive_base_cv_mfs, d1_naive_inh_mfs_cv, d1_naive_ex_mfs_cv, d1_naive_ne_mfs_cv = (
            analyze_data.get_modulation_factors(d1_naive_short[d1_naive_short["neural_response"] != "complete inhibition"],"cv")
        )

        print("\nD1-ctl modulation factors:",file=f)
        print(
            f"\tInh MFs (n={len(d1_naive_inh_mfs)}): {np.round(np.mean(d1_naive_inh_mfs),decimals)} +\\- {np.round(np.std(d1_naive_inh_mfs),decimals)}"
        ,file=f)
        print(
            f"\tExc MFs (n={len(d1_naive_ex_mfs)}): {np.round(np.mean(d1_naive_ex_mfs),decimals)} +\\- {np.round(np.std(d1_naive_ex_mfs),decimals)}"
        ,file=f)
        print(
            f"\tNE MFs (n={len(d1_naive_ne_mfs)}): {np.round(np.mean(d1_naive_ne_mfs),decimals)} +\\- {np.round(np.std(d1_naive_ne_mfs),decimals)}"
        ,file=f)
        print("\nD1-ctl linear regression Baseline FR vs MFs:",file=f)
        m, b, rval, pval, stderr = stats.linregress(
            d1_naive_base_FR_mfs[:, 1], d1_naive_base_FR_mfs[:, 0]
        )
        print(f"{pval_string(pval)}\tpval: {pval}\t Correlation Coefficient, r = {rval}",file=f)

        d1_naive_locs = analyze_data.get_d1_locations(d1_naive_short)
        print("\nD1-ctl location M-C-L (chi^2):",file=f)
        chi2_all_pval = compare_locs(d1_naive_locs)
        m_vs_c_pval = compare_locs_between(d1_naive_locs[0, :], d1_naive_locs[1, :])
        m_vs_l_pval = compare_locs_between(d1_naive_locs[0, :], d1_naive_locs[2, :])
        c_vs_l_pval = compare_locs_between(d1_naive_locs[1, :], d1_naive_locs[2, :])
        print(f"{pval_string(chi2_all_pval)}\tpval: {chi2_all_pval}\tAll Contingency Table",file=f)
        print(f"{pval_string(m_vs_c_pval)}\tpval: {m_vs_c_pval}\tMedial vs Central",file=f)
        print(f"{pval_string(m_vs_l_pval)}\tpval: {m_vs_l_pval}\tMedial vs Lateral",file=f)
        print(f"{pval_string(c_vs_l_pval)}\tpval: {c_vs_l_pval}\tCentral vs Lateral",file=f)
        print("##################################\n\n",file=f)

        print("######### Figure 2 Stats #########",file=f)
        print("GPe-ctl vs D1-ctl Norm Stim FR:",file=f)
        _, gpe_naive_norm_frs = analyze_data.get_norm_frs(gpe_naive_short)
        pval = compare_dists_ttest(gpe_naive_norm_frs, d1_naive_norm_frs)
        print(
            f"{pval_string(pval,paired=False)}\tpval: {pval}\tGPe-ctl mean: {np.round(np.mean(gpe_naive_norm_frs),decimals)} +\\- {np.round(np.std(gpe_naive_norm_frs),decimals)}\t D1-ctl mean:{np.round(np.mean(d1_naive_norm_frs),decimals)} +\\- {np.round(np.std(d1_naive_norm_frs),decimals)}"
        ,file=f)

        print("\nGPe-ctl pre-stim FR vs stim FR:",file=f)
        compare_stat_prestim_stim(
            gpe_naive_short, "baseline_freq", "stim_freq", decimals=decimals
        ,file=f)
        print("\nGPe-ctl pre-stim CV vs stim CV:",file=f)
        compare_stat_prestim_stim(
            gpe_naive_short[gpe_naive_short["neural_response"] != "complete inhibition"],
            "baseline_cv",
            "stim_cv",
            decimals=decimals,
        file=f)

        print("\nGPe-ctl vs D1-ctl response distribution (chi^2):",file=f)
        gpe_naive_responses = analyze_data.get_response_distribution(gpe_naive_short)
        pval_resp_all = compare_response_dist(d1_naive_responses, gpe_naive_responses)
        print(
            f"{pval_string(pval_resp_all)}\tpval: {pval_resp_all}\t Distribution comparison"
        ,file=f)

        pvals_resp_gpe_d1 = compare_responses(d1_naive_responses, gpe_naive_responses)
        print("\nGPe-ctl vs D1-ctl individual response comparison (fisher exact):",file=f)
        print(
            f"{pval_string(pvals_resp_gpe_d1[0])}\tpval: {pvals_resp_gpe_d1[0]}\t No Effects"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_gpe_d1[1])}\tpval: {pvals_resp_gpe_d1[1]}\t Complete Inhibitions"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_gpe_d1[2])}\tpval: {pvals_resp_gpe_d1[2]}\t Adapting Inhibitions"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_gpe_d1[3])}\tpval: {pvals_resp_gpe_d1[3]}\t Partial Inhibitions"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_gpe_d1[4])}\tpval: {pvals_resp_gpe_d1[4]}\t Excitations"
        ,file=f)

        gpe_naive_base_FR_mfs, gpe_naive_inh_mfs, gpe_naive_ex_mfs, gpe_naive_ne_mfs = (
            analyze_data.get_modulation_factors(gpe_naive_short,"frequency")
        )

        gpe_naive_base_cv_mfs, gpe_naive_inh_mfs_cv, gpe_naive_ex_mfs_cv, gpe_naive_ne_mfs_cv = (
            analyze_data.get_modulation_factors(gpe_naive_short[gpe_naive_short["neural_response"] != "complete inhibition"],"cv")
        )

        print("\nGPe-ctl modulation factors:",file=f)
        print(
            f"\tInh MFs (n={len(gpe_naive_inh_mfs)}): {np.round(np.mean(gpe_naive_inh_mfs),decimals)} +\\- {np.round(np.std(gpe_naive_inh_mfs),decimals)}"
        ,file=f)
        print(
            f"\tExc MFs (n={len(gpe_naive_ex_mfs)}): {np.round(np.mean(gpe_naive_ex_mfs),decimals)} +\\- {np.round(np.std(gpe_naive_ex_mfs),decimals)}"
        ,file=f)
        print(
            f"\tNE MFs (n={len(gpe_naive_ne_mfs)}): {np.round(np.mean(gpe_naive_ne_mfs),decimals)} +\\- {np.round(np.std(gpe_naive_ne_mfs),decimals)}"
        ,file=f)
        print("\nGPe-ctl linear regression Baseline FR vs MFs:",file=f)
        m, b, rval, pval, stderr = stats.linregress(
            gpe_naive_base_FR_mfs[:, 1], gpe_naive_base_FR_mfs[:, 0]
        )
        print(f"{pval_string(pval)}\tpval: {pval}\t Correlation Coefficient, r = {rval}",file=f)

        gpe_naive_locs = analyze_data.get_gpe_locations(gpe_naive_short)
        print("\nGPe-ctl location M-C-L (chi^2):",file=f)
        chi2_all_pval = compare_locs(gpe_naive_locs)
        m_vs_c_pval = compare_locs_between(gpe_naive_locs[0, :], gpe_naive_locs[1, :])
        m_vs_l_pval = compare_locs_between(gpe_naive_locs[0, :], gpe_naive_locs[2, :])
        c_vs_l_pval = compare_locs_between(gpe_naive_locs[1, :], gpe_naive_locs[2, :])
        print(f"{pval_string(chi2_all_pval)}\tpval: {chi2_all_pval}\tAll Contingency Table",file=f)
        print(f"{pval_string(m_vs_c_pval)}\tpval: {m_vs_c_pval}\tMedial vs Central",file=f)
        print(f"{pval_string(m_vs_l_pval)}\tpval: {m_vs_l_pval}\tMedial vs Lateral",file=f)
        print(f"{pval_string(c_vs_l_pval)}\tpval: {c_vs_l_pval}\tCentral vs Lateral",file=f)

        print("\nGPe-ctl vs D1-ctl location (chi^2):",file=f)
        m_vs_m_pval = compare_locs_between(gpe_naive_locs[0, :], d1_naive_locs[0, :])
        c_vs_c_pval = compare_locs_between(gpe_naive_locs[1, :], d1_naive_locs[1, :])
        l_vs_l_pval = compare_locs_between(gpe_naive_locs[2, :], d1_naive_locs[2, :])
        print(f"{pval_string(m_vs_m_pval)}\tpval: {m_vs_m_pval}\tMedial comparison",file=f)
        print(f"{pval_string(c_vs_c_pval)}\tpval: {c_vs_c_pval}\tCentral comparison",file=f)
        print(f"{pval_string(l_vs_l_pval)}\tpval: {l_vs_l_pval}\tLateral comparison",file=f)

        print("##################################\n\n",file=f)

        print("######### Figure 3 Stats #########",file=f)
        print("D1-ctl vs D1-DD Norm Stim FR:",file=f)
        _, d1_dd_norm_frs = analyze_data.get_norm_frs(d1_dd_short)
        pval = compare_dists_ttest(d1_dd_norm_frs, d1_naive_norm_frs)
        print(
            f"{pval_string(pval,paired=False)}\tpval: {pval}\tD1-DD mean: {np.round(np.mean(d1_dd_norm_frs),decimals)} +\\- {np.round(np.std(d1_dd_norm_frs),decimals)}\t D1-ctl mean:{np.round(np.mean(d1_naive_norm_frs),decimals)} +\\- {np.round(np.std(d1_naive_norm_frs),decimals)}"
        ,file=f)

        print("\nD1-DD pre-stim FR vs stim FR:",file=f)
        compare_stat_prestim_stim(
            d1_dd_short, "baseline_freq", "stim_freq", decimals=decimals,file=f
        )
        print("\nD1-DD pre-stim CV vs stim CV:",file=f)
        compare_stat_prestim_stim(
            d1_dd_short[d1_dd_short["neural_response"] != "complete inhibition"],
            "baseline_cv",
            "stim_cv",
            decimals=decimals,file=f
        )

        print("\nD1-DD delta FR vs D1-ctl delta FR:",file=f)
        d1_naive_delta_frs = analyze_data.get_delta_category(d1_naive_short, "freq")
        d1_dd_delta_frs = analyze_data.get_delta_category(d1_dd_short, "freq")
        pval_delta_fr = compare_dists_ttest(d1_dd_delta_frs, d1_naive_delta_frs)
        print(
            f"{pval_string(pval_delta_fr)}\tpval: {pval_delta_fr}\t D1-ctl delta frs: {np.round(np.mean(d1_naive_delta_frs),decimals)} +\\- {np.round(np.std(d1_naive_delta_frs),decimals)}\t D1-DD delta frs: {np.round(np.mean(d1_dd_delta_frs),decimals)} +\\- {np.round(np.std(d1_dd_delta_frs),decimals)}"
        ,file=f)

        print("\nD1-DD delta FR vs D1-ctl delta CV:",file=f)
        d1_naive_delta_cvs = analyze_data.get_delta_category(
            d1_naive_short[d1_naive_short["neural_response"] != "complete inhibition"], "cv"
        )
        d1_dd_delta_cvs = analyze_data.get_delta_category(
            d1_dd_short[d1_dd_short["neural_response"] != "complete inhibition"], "cv"
        )
        pval_delta_cv = compare_dists_ttest(d1_dd_delta_cvs, d1_naive_delta_cvs)
        print(
            f"{pval_string(pval_delta_cv)}\tpval: {pval_delta_cv}\t D1-ctl delta cvs: {np.round(np.mean(d1_naive_delta_cvs),decimals)} +\\- {np.round(np.std(d1_naive_delta_cvs),decimals)}\t D1-DD delta frs: {np.round(np.mean(d1_dd_delta_cvs),decimals)} +\\- {np.round(np.std(d1_dd_delta_cvs),decimals)}"
        ,file=f)

        print("\nD1-DD pre-stim CV vs stim CV:",file=f)
        compare_stat_prestim_stim(
            d1_dd_short[d1_dd_short["neural_response"] != "complete inhibition"],
            "baseline_cv",
            "stim_cv",
            decimals=decimals,file=f
        )

        print("\nD1-DD vs D1-ctl response distribution (chi^2):",file=f)
        d1_dd_responses = analyze_data.get_response_distribution(d1_dd_short)
        pval_resp_all = compare_response_dist(d1_naive_responses, d1_dd_responses)
        print(
            f"{pval_string(pval_resp_all)}\tpval: {pval_resp_all}\t Distribution comparison",file=f
        )

        pvals_resp_d1_naive_vs_dd = compare_responses(d1_naive_responses, d1_dd_responses)
        print("\nD1-DD vs D1-ctl individual response comparison (fisher exact):",file=f)
        print(
            f"{pval_string(pvals_resp_d1_naive_vs_dd[0])}\tpval: {pvals_resp_d1_naive_vs_dd[0]}\t No Effects"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_d1_naive_vs_dd[1])}\tpval: {pvals_resp_d1_naive_vs_dd[1]}\t Complete Inhibitions"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_d1_naive_vs_dd[2])}\tpval: {pvals_resp_d1_naive_vs_dd[2]}\t Adapting Inhibitions"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_d1_naive_vs_dd[3])}\tpval: {pvals_resp_d1_naive_vs_dd[3]}\t Partial Inhibitions"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_d1_naive_vs_dd[4])}\tpval: {pvals_resp_d1_naive_vs_dd[4]}\t Excitations"
        ,file=f)

        d1_dd_base_FR_mfs, d1_dd_inh_mfs, d1_dd_ex_mfs, d1_dd_ne_mfs = (
            analyze_data.get_modulation_factors(d1_dd_short,"frequency")
        ) 
        d1_dd_base_cv_mfs, d1_dd_inh_mfs_cv, d1_dd_ex_mfs_cv, d1_dd_ne_mfs_cv = (
            analyze_data.get_modulation_factors(d1_dd_short[d1_dd_short["neural_response"] != "complete inhibition"],"cv")
        ) 

        print("\nD1-DD modulation factors:",file=f)
        print(
            f"\tInh MFs (n={len(d1_dd_inh_mfs)}): {np.round(np.mean(d1_dd_inh_mfs),decimals)} +\\- {np.round(np.std(d1_dd_inh_mfs),decimals)}"
        ,file=f)
        print(
            f"\tExc MFs (n={len(d1_dd_ex_mfs)}): {np.round(np.mean(d1_dd_ex_mfs),decimals)} +\\- {np.round(np.std(d1_dd_ex_mfs),decimals)}"
        ,file=f)
        print(
            f"\tNE MFs (n={len(d1_dd_ne_mfs)}): {np.round(np.mean(d1_dd_ne_mfs),decimals)} +\\- {np.round(np.std(d1_dd_ne_mfs),decimals)}"
        ,file=f)

        print("\nD1-ctl vs D1-DD MFs FR:",file=f)
        pval = compare_dists_ttest(d1_naive_base_FR_mfs[:,1],d1_dd_base_FR_mfs[:,1])
        print(
            f"{pval_string(pval)}\tpval: {pval}\t D1-ctl MFs: {np.round(np.mean(d1_naive_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_naive_base_FR_mfs[:,1]),decimals)}\t D1-DD MFs: {np.round(np.mean(d1_dd_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_dd_base_FR_mfs[:,1]),decimals)}"
        ,file=f)

        print("\nD1-ctl vs D1-DD MFs CV:",file=f)
        pval = compare_dists_ttest(d1_naive_base_cv_mfs[:,1],d1_dd_base_cv_mfs[:,1])
        print(
            f"{pval_string(pval)}\tpval: {pval}\t D1-ctl MFs: {np.round(np.mean(d1_naive_base_cv_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_naive_base_cv_mfs[:,1]),decimals)}\t D1-DD MFs: {np.round(np.mean(d1_dd_base_cv_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_dd_base_cv_mfs[:,1]),decimals)}"
        ,file=f)

        print("\nD1-DD linear regression Baseline FR vs MFs:",file=f)
        m, b, rval, pval, stderr = stats.linregress(
            d1_dd_base_FR_mfs[:, 1], d1_dd_base_FR_mfs[:, 0]
        )
        print(f"{pval_string(pval)}\tpval: {pval}\t Correlation Coefficient, r = {rval}",file=f)

        d1_dd_locs = analyze_data.get_d1_locations(d1_dd_short)
        print("\nD1-DD location M-C-L (chi^2):",file=f)
        chi2_all_pval = compare_locs(d1_dd_locs)
        m_vs_c_pval = compare_locs_between(d1_dd_locs[0, :], d1_dd_locs[1, :])
        m_vs_l_pval = compare_locs_between(d1_dd_locs[0, :], d1_dd_locs[2, :])
        c_vs_l_pval = compare_locs_between(d1_dd_locs[1, :], d1_dd_locs[2, :])
        print(f"{pval_string(chi2_all_pval)}\tpval: {chi2_all_pval}\tAll Contingency Table",file=f)
        print(f"{pval_string(m_vs_c_pval)}\tpval: {m_vs_c_pval}\tMedial vs Central",file=f)
        print(f"{pval_string(m_vs_l_pval)}\tpval: {m_vs_l_pval}\tMedial vs Lateral",file=f)
        print(f"{pval_string(c_vs_l_pval)}\tpval: {c_vs_l_pval}\tCentral vs Lateral",file=f)

        print("\nD1-DD vs D1-ctl location (chi^2):",file=f)
        m_vs_m_pval = compare_locs_between(d1_dd_locs[0, :], d1_naive_locs[0, :])
        c_vs_c_pval = compare_locs_between(d1_dd_locs[1, :], d1_naive_locs[1, :])
        l_vs_l_pval = compare_locs_between(d1_dd_locs[2, :], d1_naive_locs[2, :])
        print(f"{pval_string(m_vs_m_pval)}\tpval: {m_vs_m_pval}\tMedial comparison",file=f)
        print(f"{pval_string(c_vs_c_pval)}\tpval: {c_vs_c_pval}\tCentral comparison",file=f)
        print(f"{pval_string(l_vs_l_pval)}\tpval: {l_vs_l_pval}\tLateral comparison",file=f)

        print("##################################\n\n",file=f)

        print("######### Figure 5 Stats #########",file=f)
        print("GPe-ctl vs GPe-DD Norm Stim FR:",file=f)
        _, gpe_dd_norm_frs = analyze_data.get_norm_frs(gpe_dd_short)
        pval = compare_dists_ttest(gpe_dd_norm_frs, gpe_naive_norm_frs)
        print(
            f"{pval_string(pval,paired=False)}\tpval: {pval}\tGPe-DD mean: {np.round(np.mean(gpe_dd_norm_frs),decimals)} +\\- {np.round(np.std(gpe_dd_norm_frs),decimals)}\t GPe-ctl mean:{np.round(np.mean(gpe_naive_norm_frs),decimals)} +\\- {np.round(np.std(gpe_naive_norm_frs),decimals)}"
        ,file=f)

        print("\nGPe-DD pre-stim FR vs stim FR:",file=f)
        compare_stat_prestim_stim(
            gpe_dd_short, "baseline_freq", "stim_freq", decimals=decimals,file=f
        )
        print("\nGPe-DD pre-stim CV vs stim CV:",file=f)
        compare_stat_prestim_stim(
            gpe_dd_short[gpe_dd_short["neural_response"] != "complete inhibition"],
            "baseline_cv",
            "stim_cv",
            decimals=decimals,file=f
        )

        print("\nGPe-DD delta FR vs GPe-ctl delta FR:",file=f)
        gpe_naive_delta_frs = analyze_data.get_delta_category(gpe_naive_short, "freq")
        gpe_dd_delta_frs = analyze_data.get_delta_category(gpe_dd_short, "freq")
        pval_delta_fr = compare_dists_ttest(gpe_dd_delta_frs, gpe_naive_delta_frs)
        print(
            f"{pval_string(pval_delta_fr)}\tpval: {pval_delta_fr}\t GPe-ctl delta frs: {np.round(np.mean(gpe_naive_delta_frs),decimals)} +\\- {np.round(np.std(gpe_naive_delta_frs),decimals)}\t GPe-DD delta frs: {np.round(np.mean(gpe_dd_delta_frs),decimals)} +\\- {np.round(np.std(gpe_dd_delta_frs),decimals)}"
        ,file=f)

        print("\nGPe-DD delta FR vs GPe-ctl delta CV:",file=f)
        gpe_naive_delta_cvs = analyze_data.get_delta_category(
            gpe_naive_short[gpe_naive_short["neural_response"] != "complete inhibition"],
            "cv",
        )
        gpe_dd_delta_cvs = analyze_data.get_delta_category(
            gpe_dd_short[gpe_dd_short["neural_response"] != "complete inhibition"], "cv"
        )
        pval_delta_cv = compare_dists_ttest(gpe_dd_delta_cvs, gpe_naive_delta_cvs)
        print(
            f"{pval_string(pval_delta_cv)}\tpval: {pval_delta_cv}\t GPe-ctl delta cvs: {np.round(np.mean(gpe_naive_delta_cvs),decimals)} +\\- {np.round(np.std(gpe_naive_delta_cvs),decimals)}\t GPe-DD delta frs: {np.round(np.mean(gpe_dd_delta_cvs),decimals)} +\\- {np.round(np.std(gpe_dd_delta_cvs),decimals)}"
        ,file=f)

        print("\nGPe-DD pre-stim CV vs stim CV:",file=f)
        compare_stat_prestim_stim(
            gpe_dd_short[gpe_dd_short["neural_response"] != "complete inhibition"],
            "baseline_cv",
            "stim_cv",
            decimals=decimals,file=f
        )

        print("\nGPe-DD vs GPe-ctl response distribution (chi^2):",file=f)
        gpe_dd_responses = analyze_data.get_response_distribution(gpe_dd_short)
        pval_resp_all = compare_response_dist(gpe_naive_responses, gpe_dd_responses)
        print(
            f"{pval_string(pval_resp_all)}\tpval: {pval_resp_all}\t Distribution comparison",file=f
        )

        pvals_resp_gpe_naive_vs_dd = compare_responses(
            gpe_naive_responses, gpe_dd_responses
        )
        print("\nGPe-DD vs GPe-ctl individual response comparison (fisher exact):",file=f)
        print(
            f"{pval_string(pvals_resp_gpe_naive_vs_dd[0])}\tpval: {pvals_resp_gpe_naive_vs_dd[0]}\t No Effects"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_gpe_naive_vs_dd[1])}\tpval: {pvals_resp_gpe_naive_vs_dd[1]}\t Complete Inhibitions"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_gpe_naive_vs_dd[2])}\tpval: {pvals_resp_gpe_naive_vs_dd[2]}\t Adapting Inhibitions"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_gpe_naive_vs_dd[3])}\tpval: {pvals_resp_gpe_naive_vs_dd[3]}\t Partial Inhibitions"
        ,file=f)
        print(
            f"{pval_string(pvals_resp_gpe_naive_vs_dd[4])}\tpval: {pvals_resp_gpe_naive_vs_dd[4]}\t Excitations"
        ,file=f)

        gpe_dd_base_FR_mfs, gpe_dd_inh_mfs, gpe_dd_ex_mfs, gpe_dd_ne_mfs = (
            analyze_data.get_modulation_factors(gpe_dd_short,"frequency")
        )

        gpe_dd_base_cv_mfs, gpe_dd_inh_mfs_cv, gpe_dd_ex_mfs_cv, gpe_dd_ne_mfs_cv = (
            analyze_data.get_modulation_factors(gpe_dd_short[gpe_dd_short["neural_response"] != "complete inhibition"],"cv")
        )  
        print("\nGPe-DD modulation factors:",file=f)
        print(
            f"\tInh MFs (n={len(gpe_dd_inh_mfs)}): {np.round(np.mean(gpe_dd_inh_mfs),decimals)} +\\- {np.round(np.std(gpe_dd_inh_mfs),decimals)}"
        ,file=f)
        print(
            f"\tExc MFs (n={len(gpe_dd_ex_mfs)}): {np.round(np.mean(gpe_dd_ex_mfs),decimals)} +\\- {np.round(np.std(gpe_dd_ex_mfs),decimals)}"
        ,file=f)
        print(
            f"\tNE MFs (n={len(gpe_dd_ne_mfs)}): {np.round(np.mean(gpe_dd_ne_mfs),decimals)} +\\- {np.round(np.std(gpe_dd_ne_mfs),decimals)}"
        ,file=f)

        print("\nGPe-ctl vs GPe-DD MFs:",file=f)
        pval = compare_dists_ttest(gpe_naive_base_FR_mfs[:,1],gpe_dd_base_FR_mfs[:,1])
        print(
            f"{pval_string(pval)}\tpval: {pval}\t GPe-ctl MFs: {np.round(np.mean(gpe_naive_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_naive_base_FR_mfs[:,1]),decimals)}\t GPe-DD MFs: {np.round(np.mean(gpe_dd_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_dd_base_FR_mfs[:,1]),decimals)}"
        ,file=f)

        print("\nGPe-ctl vs GPe-DD MFs CV:",file=f)
        pval = compare_dists_ttest(gpe_naive_base_cv_mfs[:,1],gpe_dd_base_cv_mfs[:,1])
        print(
            f"{pval_string(pval)}\tpval: {pval}\t GPe-ctl MFs: {np.round(np.mean(gpe_naive_base_cv_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_naive_base_cv_mfs[:,1]),decimals)}\t GPe-DD MFs: {np.round(np.mean(gpe_dd_base_cv_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_dd_base_cv_mfs[:,1]),decimals)}"
        ,file=f)

        print("\nGPe-DD linear regression Baseline FR vs MFs:",file=f)
        m, b, rval, pval, stderr = stats.linregress(
            gpe_dd_base_FR_mfs[:, 1], gpe_dd_base_FR_mfs[:, 0]
        )
        print(f"{pval_string(pval)}\tpval: {pval}\t Correlation Coefficient, r = {rval}",file=f)

        gpe_dd_locs = analyze_data.get_gpe_locations(gpe_dd_short)
        print("\nGPe-DD location M-C-L (chi^2):",file=f)
        chi2_all_pval = compare_locs(gpe_dd_locs)
        m_vs_c_pval = compare_locs_between(gpe_dd_locs[0, :], gpe_dd_locs[1, :])
        m_vs_l_pval = compare_locs_between(gpe_dd_locs[0, :], gpe_dd_locs[2, :])
        c_vs_l_pval = compare_locs_between(gpe_dd_locs[1, :], gpe_dd_locs[2, :])
        print(f"{pval_string(chi2_all_pval)}\tpval: {chi2_all_pval}\tAll Contingency Table",file=f)
        print(f"{pval_string(m_vs_c_pval)}\tpval: {m_vs_c_pval}\tMedial vs Central",file=f)
        print(f"{pval_string(m_vs_l_pval)}\tpval: {m_vs_l_pval}\tMedial vs Lateral",file=f)
        print(f"{pval_string(c_vs_l_pval)}\tpval: {c_vs_l_pval}\tCentral vs Lateral",file=f)

        print("\nGPe-DD vs GPe-ctl location (chi^2):",file=f)
        m_vs_m_pval = compare_locs_between(gpe_dd_locs[0, :], gpe_naive_locs[0, :])
        c_vs_c_pval = compare_locs_between(gpe_dd_locs[1, :], gpe_naive_locs[1, :])
        l_vs_l_pval = compare_locs_between(gpe_dd_locs[2, :], gpe_naive_locs[2, :])
        print(f"{pval_string(m_vs_m_pval)}\tpval: {m_vs_m_pval}\tMedial comparison",file=f)
        print(f"{pval_string(c_vs_c_pval)}\tpval: {c_vs_c_pval}\tCentral comparison",file=f)
        print(f"{pval_string(l_vs_l_pval)}\tpval: {l_vs_l_pval}\tLateral comparison",file=f)

        print("##################################\n\n",file=f)

        print("######### Figure 7 Stats #########",file=f)
        print("D1-ctl vs GPe-ctl Norm Stim FR:",file=f)
        pval = compare_dists_ttest(d1_naive_norm_frs, gpe_naive_norm_frs)
        print(
            f"{pval_string(pval,paired=False)}\tpval: {pval}\tD1-ctl mean: {np.round(np.mean(d1_naive_norm_frs),decimals)} +\\- {np.round(np.std(d1_naive_norm_frs),decimals)}\t GPe-ctl mean:{np.round(np.mean(gpe_naive_norm_frs),decimals)} +\\- {np.round(np.std(gpe_naive_norm_frs),decimals)}"
        ,file=f)

        print("\nD1-ctl vs GPe-ctl MFs:",file=f)
        pval = compare_dists_ttest(d1_naive_base_FR_mfs[:,1],gpe_naive_base_FR_mfs[:,1])
        print(
            f"{pval_string(pval)}\tpval: {pval}\t D1-ctl MFs: {np.round(np.mean(d1_naive_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_naive_base_FR_mfs[:,1]),decimals)}\t GPe-ctl MFs: {np.round(np.mean(gpe_naive_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_naive_base_FR_mfs[:,1]),decimals)}"
        ,file=f)

        print("\nD1-ctl vs GPe-ctl delta FR:",file=f)
        pval_delta_fr = compare_dists_ttest(d1_naive_delta_frs, gpe_naive_delta_frs)
        print(
            f"{pval_string(pval_delta_fr)}\tpval: {pval_delta_fr}\t D1-ctl delta frs: {np.round(np.mean(d1_naive_delta_frs),decimals)} +\\- {np.round(np.std(d1_naive_delta_frs),decimals)}\t GPe-ctl delta frs: {np.round(np.mean(gpe_naive_delta_frs),decimals)} +\\- {np.round(np.std(gpe_naive_delta_frs),decimals)}"
        ,file=f)

        print("\nD1-DD vs GPe-DD Norm Stim FR:",file=f)
        pval = compare_dists_ttest(d1_dd_norm_frs, gpe_dd_norm_frs)
        print(
            f"{pval_string(pval,paired=False)}\tpval: {pval}\tD1-DD mean: {np.round(np.mean(d1_dd_norm_frs),decimals)} +\\- {np.round(np.std(d1_dd_norm_frs),decimals)}\t GPe-DD mean:{np.round(np.mean(gpe_dd_norm_frs),decimals)} +\\- {np.round(np.std(gpe_dd_norm_frs),decimals)}"
        ,file=f)

        print("\nD1-DD vs GPe-DD MFs:",file=f)
        pval = compare_dists_ttest(d1_dd_base_FR_mfs[:,1],gpe_dd_base_FR_mfs[:,1])
        print(
            f"{pval_string(pval)}\tpval: {pval}\t D1-DD MFs: {np.round(np.mean(d1_dd_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_dd_base_FR_mfs[:,1]),decimals)}\t GPe-DD MFs: {np.round(np.mean(gpe_dd_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_dd_base_FR_mfs[:,1]),decimals)}"
        ,file=f)

        print("\nD1-DD vs GPe-DD delta FR:",file=f)
        pval_delta_fr = compare_dists_ttest(d1_dd_delta_frs, gpe_dd_delta_frs)
        print(
            f"{pval_string(pval_delta_fr)}\tpval: {pval_delta_fr}\t D1-DD delta frs: {np.round(np.mean(d1_dd_delta_frs),decimals)} +\\- {np.round(np.std(d1_dd_delta_frs),decimals)}\t GPe-DD delta frs: {np.round(np.mean(gpe_dd_delta_frs),decimals)} +\\- {np.round(np.std(gpe_dd_delta_frs),decimals)}"
        ,file=f)
        print("##################################\n\n",file=f)

        print("####### Non-figure Stats #########",file=f)
        all_naive = pd.concat([d1_naive_short,gpe_naive_short])
        all_dd = pd.concat([d1_dd_short,gpe_dd_short])

        print("\nD1-ctl vs D1-DD baseline FR:",file=f)
        pval_d1_baseline_frs = compare_dists_ttest(d1_naive_short["pre_exp_freq"],d1_dd_short["pre_exp_freq"])
        print(
            f"{pval_string(pval_d1_baseline_frs)}\tpval: {pval_d1_baseline_frs}\t D1-ctl baseline frs: {np.round(np.mean(d1_naive_short["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(d1_naive_short["pre_exp_freq"]),decimals)}\t D1-DD delta frs: {np.round(np.mean(d1_dd_short["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(d1_dd_short["pre_exp_freq"]),decimals)}"
        ,file=f)

        print("\nGPe-ctl vs GPe-DD baseline FR:",file=f)
        pval_gpe_baseline_frs = compare_dists_ttest(gpe_naive_short["pre_exp_freq"],gpe_dd_short["pre_exp_freq"])
        print(
            f"{pval_string(pval_gpe_baseline_frs)}\tpval: {pval_gpe_baseline_frs}\t GPe-ctl baseline frs: {np.round(np.mean(gpe_naive_short["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(gpe_naive_short["pre_exp_freq"]),decimals)}\t GPe-DD delta frs: {np.round(np.mean(gpe_dd_short["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(gpe_dd_short["pre_exp_freq"]),decimals)}"
        ,file=f)

        print("\nAll-ctl vs All-DD baseline FR:",file=f)
        pval_all_baseline_frs = compare_dists_ttest(all_naive["pre_exp_freq"],all_dd["pre_exp_freq"])
        print(
            f"{pval_string(pval_all_baseline_frs)}\tpval: {pval_all_baseline_frs}\t All-ctl baseline frs: {np.round(np.mean(all_naive["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(all_naive["pre_exp_freq"]),decimals)}\t All-DD delta frs: {np.round(np.mean(all_dd["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(all_dd["pre_exp_freq"]),decimals)}"
        ,file=f)

        print("\nD1-ctl vs D1-DD baseline CV:",file=f)
        pval_d1_baseline_cvs = compare_dists_ttest(d1_naive_short["pre_exp_cv"],d1_dd_short["pre_exp_cv"])
        print(
            f"{pval_string(pval_d1_baseline_cvs)}\tpval: {pval_d1_baseline_cvs}\t D1-ctl baseline cvs: {np.round(np.mean(d1_naive_short["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(d1_naive_short["pre_exp_cv"]),decimals)}\t D1-DD delta cvs: {np.round(np.mean(d1_dd_short["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(d1_dd_short["pre_exp_cv"]),decimals)}"
        ,file=f)
        print("\nGPe-ctl vs GPe-DD baseline CV:",file=f)
        pval_gpe_baseline_cvs = compare_dists_ttest(gpe_naive_short["pre_exp_cv"],gpe_dd_short["pre_exp_cv"])
        print(
            f"{pval_string(pval_gpe_baseline_cvs)}\tpval: {pval_gpe_baseline_cvs}\t GPe-ctl baseline cvs: {np.round(np.mean(gpe_naive_short["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(gpe_naive_short["pre_exp_cv"]),decimals)}\t GPe-DD delta cvs: {np.round(np.mean(gpe_dd_short["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(gpe_dd_short["pre_exp_cv"]),decimals)}"
        ,file=f)

        print("\nAll-ctl vs All-DD baseline CV:",file=f)
        pval_all_baseline_cvs = compare_dists_ttest(all_naive["pre_exp_cv"],all_dd["pre_exp_cv"])
        print(
            f"{pval_string(pval_all_baseline_cvs)}\tpval: {pval_all_baseline_cvs}\t All-ctl baseline cvs: {np.round(np.mean(all_naive["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(all_naive["pre_exp_cv"]),decimals)}\t All-DD delta cvs: {np.round(np.mean(all_dd["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(all_dd["pre_exp_cv"]),decimals)}"
        ,file=f)
        print("##################################\n\n",file=f)
