import numpy as np
import scipy.stats as stats
import pandas as pd
import analyze_data


def compare_responses(x1, x2):
    pvals = [
        stats.fisher_exact(
            [[x1[i], np.sum(x1) - x1[i]], [x2[i], np.sum(x2) - x2[i]]]
        ).pvalue
        for i in range(len(x1))
    ]
    return pvals


def compare_response_dist(x1, x2):
    return stats.chi2_contingency(np.asarray([x1, x2])).pvalue

def compare_locs(x1):
    return stats.chi2_contingency(x1).pvalue


def compare_locs_between(x1, x2):
    return stats.chi2_contingency(np.asarray([x1, x2])).pvalue


def pval_string(pval, paired=False):
    if pval < 0.001:
        return "+++" if paired else "***"
    elif pval < 0.01:
        return "++" if paired else "**"
    elif pval < 0.05:
        return "+" if paired else "*"
    else:
        return "n.s."

def compare_stat_prestim_stim(data, category1, category2, decimals=2):
    _, pval = stats.ttest_rel(data[category1], data[category2])
    print(
        f"{pval_string(pval,paired=True)}\tpval: {pval}\tPre-Stim mean: {np.round(np.mean(data[category1]),decimals)} +\\- {np.round(np.std(data[category1]),decimals)}\t Stim mean:{np.round(np.mean(data[category2]),decimals)} +\\- {np.round(np.std(data[category2]),decimals)}"
    )


def compare_dists_ttest(data1, data2):
    x1 = np.asarray(data1)
    x2 = np.asarray(data2)
    x1 = x1[np.logical_not(np.isnan(x1))]
    x2 = x2[np.logical_not(np.isnan(x2))]
    _, pval = stats.ttest_ind(
        x1, x2, alternative="less" if np.mean(x1) < np.mean(x2) else "greater"
    )
    return pval


def print_statistics(
    d1_naive_short, gpe_naive_short, d1_dd_short, gpe_dd_short, decimals=4
):
    print("######### Figure 1 Stats #########")
    print("D1-ctl pre-stim FR vs stim FR:")
    _, d1_naive_norm_frs = analyze_data.get_norm_frs(d1_naive_short)
    compare_stat_prestim_stim(
        d1_naive_short, "baseline_freq", "stim_freq", decimals=decimals
    )
    print("\nD1-ctl pre-stim CV vs stim CV:")
    compare_stat_prestim_stim(
        d1_naive_short[d1_naive_short["neural_response"] != "complete inhibition"],
        "baseline_cv",
        "stim_cv",
        decimals=decimals,
    )

    d1_naive_responses = analyze_data.get_response_distribution(d1_naive_short)
    d1_naive_base_FR_mfs, d1_naive_inh_mfs, d1_naive_ex_mfs, d1_naive_ne_mfs = (
        analyze_data.get_modulation_factors(d1_naive_short,"frequency")
    )

    d1_naive_base_cv_mfs, d1_naive_inh_mfs_cv, d1_naive_ex_mfs_cv, d1_naive_ne_mfs_cv = (
        analyze_data.get_modulation_factors(d1_naive_short[d1_naive_short["neural_response"] != "complete inhibition"],"cv")
    )

    

    print("\nD1-ctl modulation factors:")
    print(
        f"\tInh MFs (n={len(d1_naive_inh_mfs)}): {np.round(np.mean(d1_naive_inh_mfs),decimals)} +\\- {np.round(np.std(d1_naive_inh_mfs),decimals)}"
    )
    print(
        f"\tExc MFs (n={len(d1_naive_ex_mfs)}): {np.round(np.mean(d1_naive_ex_mfs),decimals)} +\\- {np.round(np.std(d1_naive_ex_mfs),decimals)}"
    )
    print(
        f"\tNE MFs (n={len(d1_naive_ne_mfs)}): {np.round(np.mean(d1_naive_ne_mfs),decimals)} +\\- {np.round(np.std(d1_naive_ne_mfs),decimals)}"
    )
    print("\nD1-ctl linear regression Baseline FR vs MFs:")
    m, b, rval, pval, stderr = stats.linregress(
        d1_naive_base_FR_mfs[:, 1], d1_naive_base_FR_mfs[:, 0]
    )
    print(f"{pval_string(pval)}\tpval: {pval}\t Correlation Coefficient, r = {rval}")

    d1_naive_locs = analyze_data.get_d1_locations(d1_naive_short)
    print("\nD1-ctl location M-C-L (chi^2):")
    chi2_all_pval = compare_locs(d1_naive_locs)
    m_vs_c_pval = compare_locs_between(d1_naive_locs[0, :], d1_naive_locs[1, :])
    m_vs_l_pval = compare_locs_between(d1_naive_locs[0, :], d1_naive_locs[2, :])
    c_vs_l_pval = compare_locs_between(d1_naive_locs[1, :], d1_naive_locs[2, :])
    print(f"{pval_string(chi2_all_pval)}\tpval: {chi2_all_pval}\tAll Contingency Table")
    print(f"{pval_string(m_vs_c_pval)}\tpval: {m_vs_c_pval}\tMedial vs Central")
    print(f"{pval_string(m_vs_l_pval)}\tpval: {m_vs_l_pval}\tMedial vs Lateral")
    print(f"{pval_string(c_vs_l_pval)}\tpval: {c_vs_l_pval}\tCentral vs Lateral")
    print("##################################\n")

    print("######### Figure 2 Stats #########")
    print("GPe-ctl vs D1-ctl Norm Stim FR:")
    _, gpe_naive_norm_frs = analyze_data.get_norm_frs(gpe_naive_short)
    pval = compare_dists_ttest(gpe_naive_norm_frs, d1_naive_norm_frs)
    print(
        f"{pval_string(pval,paired=False)}\tpval: {pval}\tGPe-ctl mean: {np.round(np.mean(gpe_naive_norm_frs),decimals)} +\\- {np.round(np.std(gpe_naive_norm_frs),decimals)}\t D1-ctl mean:{np.round(np.mean(d1_naive_norm_frs),decimals)} +\\- {np.round(np.std(d1_naive_norm_frs),decimals)}"
    )

    print("\nGPe-ctl pre-stim FR vs stim FR:")
    compare_stat_prestim_stim(
        gpe_naive_short, "baseline_freq", "stim_freq", decimals=decimals
    )
    print("\nGPe-ctl pre-stim CV vs stim CV:")
    compare_stat_prestim_stim(
        gpe_naive_short[gpe_naive_short["neural_response"] != "complete inhibition"],
        "baseline_cv",
        "stim_cv",
        decimals=decimals,
    )

    print("\nGPe-ctl vs D1-ctl response distribution (chi^2):")
    gpe_naive_responses = analyze_data.get_response_distribution(gpe_naive_short)
    pval_resp_all = compare_response_dist(d1_naive_responses, gpe_naive_responses)
    print(
        f"{pval_string(pval_resp_all)}\tpval: {pval_resp_all}\t Distribution comparison"
    )

    pvals_resp_gpe_d1 = compare_responses(d1_naive_responses, gpe_naive_responses)
    print("\nGPe-ctl vs D1-ctl individual response comparison (fisher exact):")
    print(
        f"{pval_string(pvals_resp_gpe_d1[0])}\tpval: {pvals_resp_gpe_d1[0]}\t No Effects"
    )
    print(
        f"{pval_string(pvals_resp_gpe_d1[1])}\tpval: {pvals_resp_gpe_d1[1]}\t Complete Inhibitions"
    )
    print(
        f"{pval_string(pvals_resp_gpe_d1[2])}\tpval: {pvals_resp_gpe_d1[2]}\t Adapting Inhibitions"
    )
    print(
        f"{pval_string(pvals_resp_gpe_d1[3])}\tpval: {pvals_resp_gpe_d1[3]}\t Partial Inhibitions"
    )
    print(
        f"{pval_string(pvals_resp_gpe_d1[4])}\tpval: {pvals_resp_gpe_d1[4]}\t Excitations"
    )

    gpe_naive_base_FR_mfs, gpe_naive_inh_mfs, gpe_naive_ex_mfs, gpe_naive_ne_mfs = (
        analyze_data.get_modulation_factors(gpe_naive_short,"frequency")
    )

    gpe_naive_base_cv_mfs, gpe_naive_inh_mfs_cv, gpe_naive_ex_mfs_cv, gpe_naive_ne_mfs_cv = (
        analyze_data.get_modulation_factors(gpe_naive_short[gpe_naive_short["neural_response"] != "complete inhibition"],"cv")
    )

    print("\nGPe-ctl modulation factors:")
    print(
        f"\tInh MFs (n={len(gpe_naive_inh_mfs)}): {np.round(np.mean(gpe_naive_inh_mfs),decimals)} +\\- {np.round(np.std(gpe_naive_inh_mfs),decimals)}"
    )
    print(
        f"\tExc MFs (n={len(gpe_naive_ex_mfs)}): {np.round(np.mean(gpe_naive_ex_mfs),decimals)} +\\- {np.round(np.std(gpe_naive_ex_mfs),decimals)}"
    )
    print(
        f"\tNE MFs (n={len(gpe_naive_ne_mfs)}): {np.round(np.mean(gpe_naive_ne_mfs),decimals)} +\\- {np.round(np.std(gpe_naive_ne_mfs),decimals)}"
    )
    print("\nGPe-ctl linear regression Baseline FR vs MFs:")
    m, b, rval, pval, stderr = stats.linregress(
        gpe_naive_base_FR_mfs[:, 1], gpe_naive_base_FR_mfs[:, 0]
    )
    print(f"{pval_string(pval)}\tpval: {pval}\t Correlation Coefficient, r = {rval}")

    gpe_naive_locs = analyze_data.get_gpe_locations(gpe_naive_short)
    print("\nGPe-ctl location M-C-L (chi^2):")
    chi2_all_pval = compare_locs(gpe_naive_locs)
    m_vs_c_pval = compare_locs_between(gpe_naive_locs[0, :], gpe_naive_locs[1, :])
    m_vs_l_pval = compare_locs_between(gpe_naive_locs[0, :], gpe_naive_locs[2, :])
    c_vs_l_pval = compare_locs_between(gpe_naive_locs[1, :], gpe_naive_locs[2, :])
    print(f"{pval_string(chi2_all_pval)}\tpval: {chi2_all_pval}\tAll Contingency Table")
    print(f"{pval_string(m_vs_c_pval)}\tpval: {m_vs_c_pval}\tMedial vs Central")
    print(f"{pval_string(m_vs_l_pval)}\tpval: {m_vs_l_pval}\tMedial vs Lateral")
    print(f"{pval_string(c_vs_l_pval)}\tpval: {c_vs_l_pval}\tCentral vs Lateral")

    print("\nGPe-ctl vs D1-ctl location (chi^2):")
    m_vs_m_pval = compare_locs_between(gpe_naive_locs[0, :], d1_naive_locs[0, :])
    c_vs_c_pval = compare_locs_between(gpe_naive_locs[1, :], d1_naive_locs[1, :])
    l_vs_l_pval = compare_locs_between(gpe_naive_locs[2, :], d1_naive_locs[2, :])
    print(f"{pval_string(m_vs_m_pval)}\tpval: {m_vs_m_pval}\tMedial comparison")
    print(f"{pval_string(c_vs_c_pval)}\tpval: {c_vs_c_pval}\tCentral comparison")
    print(f"{pval_string(l_vs_l_pval)}\tpval: {l_vs_l_pval}\tLateral comparison")

    print("##################################\n")

    print("######### Figure 3 Stats #########")
    print("D1-ctl vs D1-DD Norm Stim FR:")
    _, d1_dd_norm_frs = analyze_data.get_norm_frs(d1_dd_short)
    pval = compare_dists_ttest(d1_dd_norm_frs, d1_naive_norm_frs)
    print(
        f"{pval_string(pval,paired=False)}\tpval: {pval}\tD1-DD mean: {np.round(np.mean(d1_dd_norm_frs),decimals)} +\\- {np.round(np.std(d1_dd_norm_frs),decimals)}\t D1-ctl mean:{np.round(np.mean(d1_naive_norm_frs),decimals)} +\\- {np.round(np.std(d1_naive_norm_frs),decimals)}"
    )

    print("\nD1-DD pre-stim FR vs stim FR:")
    compare_stat_prestim_stim(
        d1_dd_short, "baseline_freq", "stim_freq", decimals=decimals
    )
    print("\nD1-DD pre-stim CV vs stim CV:")
    compare_stat_prestim_stim(
        d1_dd_short[d1_dd_short["neural_response"] != "complete inhibition"],
        "baseline_cv",
        "stim_cv",
        decimals=decimals,
    )

    print("\nD1-DD delta FR vs D1-ctl delta FR:")
    d1_naive_delta_frs = analyze_data.get_delta_category(d1_naive_short, "freq")
    d1_dd_delta_frs = analyze_data.get_delta_category(d1_dd_short, "freq")
    pval_delta_fr = compare_dists_ttest(d1_dd_delta_frs, d1_naive_delta_frs)
    print(
        f"{pval_string(pval_delta_fr)}\tpval: {pval_delta_fr}\t D1-ctl delta frs: {np.round(np.mean(d1_naive_delta_frs),decimals)} +\\- {np.round(np.std(d1_naive_delta_frs),decimals)}\t D1-DD delta frs: {np.round(np.mean(d1_dd_delta_frs),decimals)} +\\- {np.round(np.std(d1_dd_delta_frs),decimals)}"
    )

    print("\nD1-DD delta FR vs D1-ctl delta CV:")
    d1_naive_delta_cvs = analyze_data.get_delta_category(
        d1_naive_short[d1_naive_short["neural_response"] != "complete inhibition"], "cv"
    )
    d1_dd_delta_cvs = analyze_data.get_delta_category(
        d1_dd_short[d1_dd_short["neural_response"] != "complete inhibition"], "cv"
    )
    pval_delta_cv = compare_dists_ttest(d1_dd_delta_cvs, d1_naive_delta_cvs)
    print(
        f"{pval_string(pval_delta_cv)}\tpval: {pval_delta_cv}\t D1-ctl delta cvs: {np.round(np.mean(d1_naive_delta_cvs),decimals)} +\\- {np.round(np.std(d1_naive_delta_cvs),decimals)}\t D1-DD delta frs: {np.round(np.mean(d1_dd_delta_cvs),decimals)} +\\- {np.round(np.std(d1_dd_delta_cvs),decimals)}"
    )

    print("\nD1-DD pre-stim CV vs stim CV:")
    compare_stat_prestim_stim(
        d1_dd_short[d1_dd_short["neural_response"] != "complete inhibition"],
        "baseline_cv",
        "stim_cv",
        decimals=decimals,
    )

    print("\nD1-DD vs D1-ctl response distribution (chi^2):")
    d1_dd_responses = analyze_data.get_response_distribution(d1_dd_short)
    pval_resp_all = compare_response_dist(d1_naive_responses, d1_dd_responses)
    print(
        f"{pval_string(pval_resp_all)}\tpval: {pval_resp_all}\t Distribution comparison"
    )

    pvals_resp_d1_naive_vs_dd = compare_responses(d1_naive_responses, d1_dd_responses)
    print("\nD1-DD vs D1-ctl individual response comparison (fisher exact):")
    print(
        f"{pval_string(pvals_resp_d1_naive_vs_dd[0])}\tpval: {pvals_resp_d1_naive_vs_dd[0]}\t No Effects"
    )
    print(
        f"{pval_string(pvals_resp_d1_naive_vs_dd[1])}\tpval: {pvals_resp_d1_naive_vs_dd[1]}\t Complete Inhibitions"
    )
    print(
        f"{pval_string(pvals_resp_d1_naive_vs_dd[2])}\tpval: {pvals_resp_d1_naive_vs_dd[2]}\t Adapting Inhibitions"
    )
    print(
        f"{pval_string(pvals_resp_d1_naive_vs_dd[3])}\tpval: {pvals_resp_d1_naive_vs_dd[3]}\t Partial Inhibitions"
    )
    print(
        f"{pval_string(pvals_resp_d1_naive_vs_dd[4])}\tpval: {pvals_resp_d1_naive_vs_dd[4]}\t Excitations"
    )

    d1_dd_base_FR_mfs, d1_dd_inh_mfs, d1_dd_ex_mfs, d1_dd_ne_mfs = (
        analyze_data.get_modulation_factors(d1_dd_short,"frequency")
    ) 
    d1_dd_base_cv_mfs, d1_dd_inh_mfs_cv, d1_dd_ex_mfs_cv, d1_dd_ne_mfs_cv = (
        analyze_data.get_modulation_factors(d1_dd_short[d1_dd_short["neural_response"] != "complete inhibition"],"cv")
    ) 

    print("\nD1-DD modulation factors:")
    print(
        f"\tInh MFs (n={len(d1_dd_inh_mfs)}): {np.round(np.mean(d1_dd_inh_mfs),decimals)} +\\- {np.round(np.std(d1_dd_inh_mfs),decimals)}"
    )
    print(
        f"\tExc MFs (n={len(d1_dd_ex_mfs)}): {np.round(np.mean(d1_dd_ex_mfs),decimals)} +\\- {np.round(np.std(d1_dd_ex_mfs),decimals)}"
    )
    print(
        f"\tNE MFs (n={len(d1_dd_ne_mfs)}): {np.round(np.mean(d1_dd_ne_mfs),decimals)} +\\- {np.round(np.std(d1_dd_ne_mfs),decimals)}"
    )

    print("\nD1-ctl vs D1-DD MFs FR:")
    pval = compare_dists_ttest(d1_naive_base_FR_mfs[:,1],d1_dd_base_FR_mfs[:,1])
    print(
        f"{pval_string(pval)}\tpval: {pval}\t D1-ctl MFs: {np.round(np.mean(d1_naive_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_naive_base_FR_mfs[:,1]),decimals)}\t D1-DD MFs: {np.round(np.mean(d1_dd_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_dd_base_FR_mfs[:,1]),decimals)}"
    )

    print("\nD1-ctl vs D1-DD MFs CV:")
    pval = compare_dists_ttest(d1_naive_base_cv_mfs[:,1],d1_dd_base_cv_mfs[:,1])
    print(
        f"{pval_string(pval)}\tpval: {pval}\t D1-ctl MFs: {np.round(np.mean(d1_naive_base_cv_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_naive_base_cv_mfs[:,1]),decimals)}\t D1-DD MFs: {np.round(np.mean(d1_dd_base_cv_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_dd_base_cv_mfs[:,1]),decimals)}"
    )

    print("\nD1-DD linear regression Baseline FR vs MFs:")
    m, b, rval, pval, stderr = stats.linregress(
        d1_dd_base_FR_mfs[:, 1], d1_dd_base_FR_mfs[:, 0]
    )
    print(f"{pval_string(pval)}\tpval: {pval}\t Correlation Coefficient, r = {rval}")

    d1_dd_locs = analyze_data.get_d1_locations(d1_dd_short)
    print("\nD1-DD location M-C-L (chi^2):")
    chi2_all_pval = compare_locs(d1_dd_locs)
    m_vs_c_pval = compare_locs_between(d1_dd_locs[0, :], d1_dd_locs[1, :])
    m_vs_l_pval = compare_locs_between(d1_dd_locs[0, :], d1_dd_locs[2, :])
    c_vs_l_pval = compare_locs_between(d1_dd_locs[1, :], d1_dd_locs[2, :])
    print(f"{pval_string(chi2_all_pval)}\tpval: {chi2_all_pval}\tAll Contingency Table")
    print(f"{pval_string(m_vs_c_pval)}\tpval: {m_vs_c_pval}\tMedial vs Central")
    print(f"{pval_string(m_vs_l_pval)}\tpval: {m_vs_l_pval}\tMedial vs Lateral")
    print(f"{pval_string(c_vs_l_pval)}\tpval: {c_vs_l_pval}\tCentral vs Lateral")

    print("\nD1-DD vs D1-ctl location (chi^2):")
    m_vs_m_pval = compare_locs_between(d1_dd_locs[0, :], d1_naive_locs[0, :])
    c_vs_c_pval = compare_locs_between(d1_dd_locs[1, :], d1_naive_locs[1, :])
    l_vs_l_pval = compare_locs_between(d1_dd_locs[2, :], d1_naive_locs[2, :])
    print(f"{pval_string(m_vs_m_pval)}\tpval: {m_vs_m_pval}\tMedial comparison")
    print(f"{pval_string(c_vs_c_pval)}\tpval: {c_vs_c_pval}\tCentral comparison")
    print(f"{pval_string(l_vs_l_pval)}\tpval: {l_vs_l_pval}\tLateral comparison")

    print("##################################\n")

    print("######### Figure 5 Stats #########")
    print("GPe-ctl vs GPe-DD Norm Stim FR:")
    _, gpe_dd_norm_frs = analyze_data.get_norm_frs(gpe_dd_short)
    pval = compare_dists_ttest(gpe_dd_norm_frs, gpe_naive_norm_frs)
    print(
        f"{pval_string(pval,paired=False)}\tpval: {pval}\tGPe-DD mean: {np.round(np.mean(gpe_dd_norm_frs),decimals)} +\\- {np.round(np.std(gpe_dd_norm_frs),decimals)}\t GPe-ctl mean:{np.round(np.mean(gpe_naive_norm_frs),decimals)} +\\- {np.round(np.std(gpe_naive_norm_frs),decimals)}"
    )

    print("\nGPe-DD pre-stim FR vs stim FR:")
    compare_stat_prestim_stim(
        gpe_dd_short, "baseline_freq", "stim_freq", decimals=decimals
    )
    print("\nGPe-DD pre-stim CV vs stim CV:")
    compare_stat_prestim_stim(
        gpe_dd_short[gpe_dd_short["neural_response"] != "complete inhibition"],
        "baseline_cv",
        "stim_cv",
        decimals=decimals,
    )

    print("\nGPe-DD delta FR vs GPe-ctl delta FR:")
    gpe_naive_delta_frs = analyze_data.get_delta_category(gpe_naive_short, "freq")
    gpe_dd_delta_frs = analyze_data.get_delta_category(gpe_dd_short, "freq")
    pval_delta_fr = compare_dists_ttest(gpe_dd_delta_frs, gpe_naive_delta_frs)
    print(
        f"{pval_string(pval_delta_fr)}\tpval: {pval_delta_fr}\t GPe-ctl delta frs: {np.round(np.mean(gpe_naive_delta_frs),decimals)} +\\- {np.round(np.std(gpe_naive_delta_frs),decimals)}\t GPe-DD delta frs: {np.round(np.mean(gpe_dd_delta_frs),decimals)} +\\- {np.round(np.std(gpe_dd_delta_frs),decimals)}"
    )

    print("\nGPe-DD delta FR vs GPe-ctl delta CV:")
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
    )

    print("\nGPe-DD pre-stim CV vs stim CV:")
    compare_stat_prestim_stim(
        gpe_dd_short[gpe_dd_short["neural_response"] != "complete inhibition"],
        "baseline_cv",
        "stim_cv",
        decimals=decimals,
    )

    print("\nGPe-DD vs GPe-ctl response distribution (chi^2):")
    gpe_dd_responses = analyze_data.get_response_distribution(gpe_dd_short)
    pval_resp_all = compare_response_dist(gpe_naive_responses, gpe_dd_responses)
    print(
        f"{pval_string(pval_resp_all)}\tpval: {pval_resp_all}\t Distribution comparison"
    )

    pvals_resp_gpe_naive_vs_dd = compare_responses(
        gpe_naive_responses, gpe_dd_responses
    )
    print("\nGPe-DD vs GPe-ctl individual response comparison (fisher exact):")
    print(
        f"{pval_string(pvals_resp_gpe_naive_vs_dd[0])}\tpval: {pvals_resp_gpe_naive_vs_dd[0]}\t No Effects"
    )
    print(
        f"{pval_string(pvals_resp_gpe_naive_vs_dd[1])}\tpval: {pvals_resp_gpe_naive_vs_dd[1]}\t Complete Inhibitions"
    )
    print(
        f"{pval_string(pvals_resp_gpe_naive_vs_dd[2])}\tpval: {pvals_resp_gpe_naive_vs_dd[2]}\t Adapting Inhibitions"
    )
    print(
        f"{pval_string(pvals_resp_gpe_naive_vs_dd[3])}\tpval: {pvals_resp_gpe_naive_vs_dd[3]}\t Partial Inhibitions"
    )
    print(
        f"{pval_string(pvals_resp_gpe_naive_vs_dd[4])}\tpval: {pvals_resp_gpe_naive_vs_dd[4]}\t Excitations"
    )

    gpe_dd_base_FR_mfs, gpe_dd_inh_mfs, gpe_dd_ex_mfs, gpe_dd_ne_mfs = (
        analyze_data.get_modulation_factors(gpe_dd_short,"frequency")
    )

    gpe_dd_base_cv_mfs, gpe_dd_inh_mfs_cv, gpe_dd_ex_mfs_cv, gpe_dd_ne_mfs_cv = (
        analyze_data.get_modulation_factors(gpe_dd_short[gpe_dd_short["neural_response"] != "complete inhibition"],"cv")
    )  
    print("\nGPe-DD modulation factors:")
    print(
        f"\tInh MFs (n={len(gpe_dd_inh_mfs)}): {np.round(np.mean(gpe_dd_inh_mfs),decimals)} +\\- {np.round(np.std(gpe_dd_inh_mfs),decimals)}"
    )
    print(
        f"\tExc MFs (n={len(gpe_dd_ex_mfs)}): {np.round(np.mean(gpe_dd_ex_mfs),decimals)} +\\- {np.round(np.std(gpe_dd_ex_mfs),decimals)}"
    )
    print(
        f"\tNE MFs (n={len(gpe_dd_ne_mfs)}): {np.round(np.mean(gpe_dd_ne_mfs),decimals)} +\\- {np.round(np.std(gpe_dd_ne_mfs),decimals)}"
    )

    print("\nGPe-ctl vs GPe-DD MFs:")
    pval = compare_dists_ttest(gpe_naive_base_FR_mfs[:,1],gpe_dd_base_FR_mfs[:,1])
    print(
        f"{pval_string(pval)}\tpval: {pval}\t GPe-ctl MFs: {np.round(np.mean(gpe_naive_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_naive_base_FR_mfs[:,1]),decimals)}\t GPe-DD MFs: {np.round(np.mean(gpe_dd_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_dd_base_FR_mfs[:,1]),decimals)}"
    )

    print("\nGPe-ctl vs GPe-DD MFs CV:")
    pval = compare_dists_ttest(gpe_naive_base_cv_mfs[:,1],gpe_dd_base_cv_mfs[:,1])
    print(
        f"{pval_string(pval)}\tpval: {pval}\t GPe-ctl MFs: {np.round(np.mean(gpe_naive_base_cv_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_naive_base_cv_mfs[:,1]),decimals)}\t GPe-DD MFs: {np.round(np.mean(gpe_dd_base_cv_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_dd_base_cv_mfs[:,1]),decimals)}"
    )

    print("\nGPe-DD linear regression Baseline FR vs MFs:")
    m, b, rval, pval, stderr = stats.linregress(
        gpe_dd_base_FR_mfs[:, 1], gpe_dd_base_FR_mfs[:, 0]
    )
    print(f"{pval_string(pval)}\tpval: {pval}\t Correlation Coefficient, r = {rval}")

    gpe_dd_locs = analyze_data.get_gpe_locations(gpe_dd_short)
    print("\nGPe-DD location M-C-L (chi^2):")
    chi2_all_pval = compare_locs(gpe_dd_locs)
    m_vs_c_pval = compare_locs_between(gpe_dd_locs[0, :], gpe_dd_locs[1, :])
    m_vs_l_pval = compare_locs_between(gpe_dd_locs[0, :], gpe_dd_locs[2, :])
    c_vs_l_pval = compare_locs_between(gpe_dd_locs[1, :], gpe_dd_locs[2, :])
    print(f"{pval_string(chi2_all_pval)}\tpval: {chi2_all_pval}\tAll Contingency Table")
    print(f"{pval_string(m_vs_c_pval)}\tpval: {m_vs_c_pval}\tMedial vs Central")
    print(f"{pval_string(m_vs_l_pval)}\tpval: {m_vs_l_pval}\tMedial vs Lateral")
    print(f"{pval_string(c_vs_l_pval)}\tpval: {c_vs_l_pval}\tCentral vs Lateral")

    print("\nGPe-DD vs GPe-ctl location (chi^2):")
    m_vs_m_pval = compare_locs_between(gpe_dd_locs[0, :], gpe_naive_locs[0, :])
    c_vs_c_pval = compare_locs_between(gpe_dd_locs[1, :], gpe_naive_locs[1, :])
    l_vs_l_pval = compare_locs_between(gpe_dd_locs[2, :], gpe_naive_locs[2, :])
    print(f"{pval_string(m_vs_m_pval)}\tpval: {m_vs_m_pval}\tMedial comparison")
    print(f"{pval_string(c_vs_c_pval)}\tpval: {c_vs_c_pval}\tCentral comparison")
    print(f"{pval_string(l_vs_l_pval)}\tpval: {l_vs_l_pval}\tLateral comparison")

    print("##################################\n")

    print("######### Figure 7 Stats #########")
    print("D1-ctl vs GPe-ctl Norm Stim FR:")
    pval = compare_dists_ttest(d1_naive_norm_frs, gpe_naive_norm_frs)
    print(
        f"{pval_string(pval,paired=False)}\tpval: {pval}\tD1-ctl mean: {np.round(np.mean(d1_naive_norm_frs),decimals)} +\\- {np.round(np.std(d1_naive_norm_frs),decimals)}\t GPe-ctl mean:{np.round(np.mean(gpe_naive_norm_frs),decimals)} +\\- {np.round(np.std(gpe_naive_norm_frs),decimals)}"
    )

    print("\nD1-ctl vs GPe-ctl MFs:")
    pval = compare_dists_ttest(d1_naive_base_FR_mfs[:,1],gpe_naive_base_FR_mfs[:,1])
    print(
        f"{pval_string(pval)}\tpval: {pval}\t D1-ctl MFs: {np.round(np.mean(d1_naive_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_naive_base_FR_mfs[:,1]),decimals)}\t GPe-ctl MFs: {np.round(np.mean(gpe_naive_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_naive_base_FR_mfs[:,1]),decimals)}"
    )

    print("\nD1-ctl vs GPe-ctl delta FR:")
    pval_delta_fr = compare_dists_ttest(d1_naive_delta_frs, gpe_naive_delta_frs)
    print(
        f"{pval_string(pval_delta_fr)}\tpval: {pval_delta_fr}\t D1-ctl delta frs: {np.round(np.mean(d1_naive_delta_frs),decimals)} +\\- {np.round(np.std(d1_naive_delta_frs),decimals)}\t GPe-ctl delta frs: {np.round(np.mean(gpe_naive_delta_frs),decimals)} +\\- {np.round(np.std(gpe_naive_delta_frs),decimals)}"
    )

    print("\nD1-DD vs GPe-DD Norm Stim FR:")
    pval = compare_dists_ttest(d1_dd_norm_frs, gpe_dd_norm_frs)
    print(
        f"{pval_string(pval,paired=False)}\tpval: {pval}\tD1-DD mean: {np.round(np.mean(d1_dd_norm_frs),decimals)} +\\- {np.round(np.std(d1_dd_norm_frs),decimals)}\t GPe-DD mean:{np.round(np.mean(gpe_dd_norm_frs),decimals)} +\\- {np.round(np.std(gpe_dd_norm_frs),decimals)}"
    )

    print("\nD1-DD vs GPe-DD MFs:")
    pval = compare_dists_ttest(d1_dd_base_FR_mfs[:,1],gpe_dd_base_FR_mfs[:,1])
    print(
        f"{pval_string(pval)}\tpval: {pval}\t D1-DD MFs: {np.round(np.mean(d1_dd_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(d1_dd_base_FR_mfs[:,1]),decimals)}\t GPe-DD MFs: {np.round(np.mean(gpe_dd_base_FR_mfs[:,1]),decimals)} +\\- {np.round(np.std(gpe_dd_base_FR_mfs[:,1]),decimals)}"
    )

    print("\nD1-DD vs GPe-DD delta FR:")
    pval_delta_fr = compare_dists_ttest(d1_dd_delta_frs, gpe_dd_delta_frs)
    print(
        f"{pval_string(pval_delta_fr)}\tpval: {pval_delta_fr}\t D1-DD delta frs: {np.round(np.mean(d1_dd_delta_frs),decimals)} +\\- {np.round(np.std(d1_dd_delta_frs),decimals)}\t GPe-DD delta frs: {np.round(np.mean(gpe_dd_delta_frs),decimals)} +\\- {np.round(np.std(gpe_dd_delta_frs),decimals)}"
    )
    print("##################################\n")

    print("####### Non-figure Stats #########")
    all_naive = pd.concat([d1_naive_short,gpe_naive_short])
    all_dd = pd.concat([d1_dd_short,gpe_dd_short])

    print("\nD1-ctl vs D1-DD baseline FR:")
    pval_d1_baseline_frs = compare_dists_ttest(d1_naive_short["pre_exp_freq"],d1_dd_short["pre_exp_freq"])
    print(
        f"{pval_string(pval_d1_baseline_frs)}\tpval: {pval_d1_baseline_frs}\t D1-ctl baseline frs: {np.round(np.mean(d1_naive_short["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(d1_naive_short["pre_exp_freq"]),decimals)}\t D1-DD delta frs: {np.round(np.mean(d1_dd_short["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(d1_dd_short["pre_exp_freq"]),decimals)}"
    )

    print("\nGPe-ctl vs GPe-DD baseline FR:")
    pval_gpe_baseline_frs = compare_dists_ttest(gpe_naive_short["pre_exp_freq"],gpe_dd_short["pre_exp_freq"])
    print(
        f"{pval_string(pval_gpe_baseline_frs)}\tpval: {pval_gpe_baseline_frs}\t GPe-ctl baseline frs: {np.round(np.mean(gpe_naive_short["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(gpe_naive_short["pre_exp_freq"]),decimals)}\t GPe-DD delta frs: {np.round(np.mean(gpe_dd_short["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(gpe_dd_short["pre_exp_freq"]),decimals)}"
    )

    print("\nAll-ctl vs All-DD baseline FR:")
    pval_all_baseline_frs = compare_dists_ttest(all_naive["pre_exp_freq"],all_dd["pre_exp_freq"])
    print(
        f"{pval_string(pval_all_baseline_frs)}\tpval: {pval_all_baseline_frs}\t All-ctl baseline frs: {np.round(np.mean(all_naive["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(all_naive["pre_exp_freq"]),decimals)}\t All-DD delta frs: {np.round(np.mean(all_dd["pre_exp_freq"]),decimals)} +\\- {np.round(np.std(all_dd["pre_exp_freq"]),decimals)}"
    )

    print("\nD1-ctl vs D1-DD baseline CV:")
    pval_d1_baseline_cvs = compare_dists_ttest(d1_naive_short["pre_exp_cv"],d1_dd_short["pre_exp_cv"])
    print(
        f"{pval_string(pval_d1_baseline_cvs)}\tpval: {pval_d1_baseline_cvs}\t D1-ctl baseline cvs: {np.round(np.mean(d1_naive_short["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(d1_naive_short["pre_exp_cv"]),decimals)}\t D1-DD delta cvs: {np.round(np.mean(d1_dd_short["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(d1_dd_short["pre_exp_cv"]),decimals)}"
    )
    print("\nGPe-ctl vs GPe-DD baseline CV:")
    pval_gpe_baseline_cvs = compare_dists_ttest(gpe_naive_short["pre_exp_cv"],gpe_dd_short["pre_exp_cv"])
    print(
        f"{pval_string(pval_gpe_baseline_cvs)}\tpval: {pval_gpe_baseline_cvs}\t GPe-ctl baseline cvs: {np.round(np.mean(gpe_naive_short["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(gpe_naive_short["pre_exp_cv"]),decimals)}\t GPe-DD delta cvs: {np.round(np.mean(gpe_dd_short["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(gpe_dd_short["pre_exp_cv"]),decimals)}"
    )

    print("\nAll-ctl vs All-DD baseline CV:")
    pval_all_baseline_cvs = compare_dists_ttest(all_naive["pre_exp_cv"],all_dd["pre_exp_cv"])
    print(
        f"{pval_string(pval_all_baseline_cvs)}\tpval: {pval_all_baseline_cvs}\t All-ctl baseline cvs: {np.round(np.mean(all_naive["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(all_naive["pre_exp_cv"]),decimals)}\t All-DD delta cvs: {np.round(np.mean(all_dd["pre_exp_cv"]),decimals)} +\\- {np.round(np.std(all_dd["pre_exp_cv"]),decimals)}"
    )
    print("##################################\n")


    print("####### Trial Average Deltas #########")
    pval = compare_dists_ttest(d1_naive_delta_frs,gpe_naive_delta_frs)
    print(
        f"{pval_string(pval)}\tpval: {pval}\t D1-ctl deltas: {np.round(np.mean(d1_naive_delta_frs),decimals)} +\\- {np.round(np.std(d1_naive_delta_frs),decimals)}\t GPe-ctl deltas: {np.round(np.mean(gpe_naive_delta_frs),decimals)} +\\- {np.round(np.std(gpe_naive_delta_frs),decimals)}"
    )

    pval = compare_dists_ttest(d1_dd_delta_frs,gpe_dd_delta_frs)
    print(
        f"{pval_string(pval)}\tpval: {pval}\t D1-DD deltas: {np.round(np.mean(d1_dd_delta_frs),decimals)} +\\- {np.round(np.std(d1_dd_delta_frs),decimals)}\t GPe-DD deltas: {np.round(np.mean(gpe_dd_delta_frs),decimals)} +\\- {np.round(np.std(gpe_dd_delta_frs),decimals)}"
    )
    print("##################################\n")

    d1_naive_all_MFs = analyze_data.get_all_modulation_factors(d1_naive_short)
    gpe_naive_all_MFs = analyze_data.get_all_modulation_factors(gpe_naive_short)
    d1_dd_all_MFs = analyze_data.get_all_modulation_factors(d1_dd_short)
    gpe_dd_all_MFs = analyze_data.get_all_modulation_factors(gpe_dd_short)

    d1_naive_all_deltas = analyze_data.get_all_deltas(d1_naive_short,"freq")
    gpe_naive_all_deltas = analyze_data.get_all_deltas(gpe_naive_short,"freq")
    d1_dd_all_deltas = analyze_data.get_all_deltas(d1_dd_short,"freq")
    gpe_dd_all_deltas = analyze_data.get_all_deltas(gpe_dd_short,"freq")


    print("####### All MFs #########")
    pval = compare_dists_ttest(d1_naive_all_MFs,gpe_naive_all_MFs)
    print(
        f"{pval_string(pval)}\tpval: {pval}\t All-D1-ctl MFs: {np.round(np.mean(d1_naive_all_MFs),decimals)} +\\- {np.round(np.std(d1_naive_all_MFs),decimals)}\t All-GPe-ctl MFs: {np.round(np.mean(gpe_naive_all_MFs),decimals)} +\\- {np.round(np.std(gpe_naive_all_MFs),decimals)}"
    )

    pval = compare_dists_ttest(d1_dd_all_MFs,gpe_dd_all_MFs)
    print(
        f"{pval_string(pval)}\tpval: {pval}\t All-D1-DD MFs: {np.round(np.mean(d1_dd_all_MFs),decimals)} +\\- {np.round(np.std(d1_dd_all_MFs),decimals)}\t All-GPe-DD MFs: {np.round(np.mean(gpe_dd_all_MFs),decimals)} +\\- {np.round(np.std(gpe_dd_all_MFs),decimals)}"
    )
    print("##################################\n")

    print("####### All Deltas #########")
    pval = compare_dists_ttest(d1_naive_all_deltas,gpe_naive_all_deltas)
    print(
        f"{pval_string(pval)}\tpval: {pval}\t All-D1-ctl deltas: {np.round(np.mean(d1_naive_all_deltas),decimals)} +\\- {np.round(np.std(d1_naive_all_deltas),decimals)}\t All-GPe-ctl deltas: {np.round(np.mean(gpe_naive_all_deltas),decimals)} +\\- {np.round(np.std(gpe_naive_all_deltas),decimals)}"
    )

    pval = compare_dists_ttest(d1_dd_all_deltas,gpe_dd_all_deltas)
    print(
        f"{pval_string(pval)}\tpval: {pval}\t All-D1-DD deltas: {np.round(np.mean(d1_dd_all_deltas),decimals)} +\\- {np.round(np.std(d1_dd_all_deltas),decimals)}\t All-GPe-DD deltas: {np.round(np.mean(gpe_dd_all_deltas),decimals)} +\\- {np.round(np.std(gpe_dd_all_deltas),decimals)}"
    )
    print("##################################\n")

    """
    pval = compare_response_dist(d1_naive_responses,d1_dd_responses)
    print(pval)

    pval = compare_response_dist(gpe_naive_responses,gpe_dd_responses)
    print(pval)

    pval = compare_response_dist(d1_naive_responses,gpe_naive_responses)
    print(pval)

    pval = compare_response_dist(d1_dd_responses,gpe_dd_responses)
    print(pval)
    """
