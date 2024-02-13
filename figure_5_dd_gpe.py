"""
File: figure_5_dd_gpe.py
Author: John E. Parker
Date: 13 February 2024

Description: Generates figure 5 based gpe dd data and compares with gpe naive data.
"""

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib import *

# import user modules
from helpers import *
from generate_figures import *


# Generate figure based on data set and comarison data set
def generate_figure(data_short, compare_short, save_dir):
    """
    Generates figure 4 with gpe dd data set, compared with gpe naive data, and displays figure.

    data_short\t:\tdataframe of gpe dd data
    compare_short\t:\tdataframe of gpe naive data for comparison
    save_dir\t:\tdirectory to save figure
    """

    # Set figure font settings
    rcParams["font.sans-serif"] = "Arial"
    rcParams["font.size"] = 8
    rcParams["axes.linewidth"] = 0.5

    # Color choices each respective data set
    d1_color = "#d8531a"
    d1_semcolor = "#e6984d"
    d1_dd_color = "#751a02"
    d1_dd_semcolor = "#9b5442"  # "#d0afa6"
    gpe_color = "#9966e6"
    gpe_semcolor = "#bf8cff"
    gpe_dd_color = "#330080"
    gpe_dd_semcolor = "#6633b3"

    # Generate figure
    fig = plt.figure(figsize=(7, 8), dpi=300, tight_layout=True)  # figsize=(12,16)
    gs = gridspec.GridSpec(3, 1, height_ratios=[4, 2, 2])  # SPLIT INTO 3 ROWS
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0], width_ratios=[1, 4]
    )

    # Keep empty section for histology
    histology = [fig.add_subplot(gs1[0])]
    histology[0].axis("off")

    # Set up section for normalized PSTH
    gs1B = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs1[1], hspace=0.05)
    psth_example = [fig.add_subplot(gs1B[0]), fig.add_subplot(gs1B[1])]

    # Values to plot
    bw = 0.05  # bin width
    baseline = 2
    stim = 1
    post = 0
    # plot range
    plot_lims = [-1, 1]

    # plot random set neuron's trial 1
    plot_random_trials(
        data_short,
        psth_example[0],
        plot_lims,
        n=len(data_short) // 4,
        baseline=baseline,
        stim=1,
        post=0,
    )

    # plot normalized PSTH relative to 1s of pre-stim firing and compare with d1 naive data
    psth_norm_compare(
        data_short,
        compare_short,
        psth_example[1],
        bw=0.05,
        baseline=1,
        stim=1,
        post=0,
        plot_lims=[-1, 1],
        color1=gpe_dd_color,
        semcolor1=gpe_dd_semcolor,
        color2="k",
        semcolor2="gray",
        drawboth=True,
        label1=f"GPe-DD ($n={len(data_short)}$)",
        label2=f"GPe-ctl ($n={len(compare_short)}$)",
        lw=1,
    )

    psth_example[1].set_ylabel("Norm. Firing Rate")

    # Plot changes in FR and CV for pre-stim vs stim periods
    gs1CD = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs[1], width_ratios=[1, 1, 1, 1], wspace=0.75
    )
    fr_cv_change = [fig.add_subplot(gs1CD[i]) for i in range(4)]

    category_shift_single(
        data_short,
        "baseline_freq",
        "stim_freq",
        fr_cv_change[0],
        color1=gpe_dd_color,
        tick1="Pre",
        tick2="Stim",
        title="Firing Rate",
        ylabel="Hz",
    )

    category_shift_single(
        data_short[data_short["neural_response"] != "complete inhibition"],
        "baseline_cv",
        "stim_cv",
        fr_cv_change[1],
        tick1="Pre",
        tick2="Stim",
        title="CV",
        ylabel="CV",
        color1=gpe_dd_color,
    )

    # Plot modulation factors for FR and CV, compare with gpe naive MFs (remove complete inhibition for CV)
    category_change_MF(
        compare_short,
        data_short,
        "baseline_freq",
        "stim_freq",
        fr_cv_change[2],
        color1=gpe_color,
        color2=gpe_dd_color,
        ylabel="Modulation Factor",
        title="$\\Delta$ Firing Rate",
        tick1="GPe-ctl",
        tick2="GPe-DD",
        ylims=[-1, 0.5],
    )

    category_change_MF(
        compare_short[compare_short["neural_response"] != "complete inhibition"],
        data_short[data_short["neural_response"] != "complete inhibition"],
        "baseline_cv",
        "stim_cv",
        fr_cv_change[3],
        color1=gpe_color,
        color2=gpe_dd_color,
        ylabel="Modulation Factor",
        title="$\\Delta$ CV",
        tick1="GPe-ctl",
        tick2="GPe-DD",
        ylims=[-0.4, 0.1],
    )

    # Set up figure section for classification results
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.25)
    classifications = [fig.add_subplot(gs3[i]) for i in range(3)]

    # Plot distribution of responses, compare with gpe naive data
    classification_plot_comparison_horizontal(
        data_short,
        compare_short,
        classifications[0],
        "GPe-DD",
        "GPe-ctl",
        plot_vals=True,
    )

    # Plot responses by location
    locations_plot_gpe(data_short, classifications[1])

    # Plot responses by modulation factor and baseline firing rate
    baseline_class_prediction_MF(data_short, classifications[2])

    # Label figure and clean up
    add_fig_labels(
        [
            histology[0],
            psth_example[0],
            fr_cv_change[0],
            fr_cv_change[2],
            classifications[0],
            classifications[1],
            classifications[2],
        ]
    )

    # Save and display figure
    plt.savefig(f"{save_dir}/figure_5.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close()
    run_cmd(f"open {save_dir}/figure_5.pdf")
