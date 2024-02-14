"""
File: figure_1_naive_d1.py
Author: John E. Parker
Date: 13 February 2024

Description: Generates figure 1 based d1 naive data.
"""

import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib import *

# import user modules
from helpers import *
from generate_figures import *


# Generate figure based on data set
def generate_figure(data_short, save_dir):
    """
    Generates figure 1 with d1 naive data set and displays figure.

    data_short\t:\tdataframe of d1 naive data
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
    d1_dd_semcolor = "#9b5442"
    gpe_color = "#9966e6"
    gpe_semcolor = "#bf8cff"
    gpe_dd_color = "#330080"
    gpe_dd_semcolor = "#6633b3"

    # Generate figure
    fig = plt.figure(figsize=(7, 8), dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 4, 2, 4])
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])

    # Keep empty section for histology
    histology = [fig.add_subplot(gs0[0])]
    histology[0].axis("off")

    # Set up section for normalized PSTH
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1], width_ratios=[5, 3]
    )
    gs1B = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs1[0], hspace=0.05)
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

    # plot normalized PSTH relative to 1s of pre-stim firing
    norm_results = psth_norm(
        data_short,
        psth_example[1],
        bw=bw,
        baseline=baseline,
        stim=stim,
        post=post,
        plot_lims=plot_lims,
        label=f"D1 Control ($n=${len(data_short)})",
        color=d1_color,
        semcolor=d1_semcolor,
        lw=1,
    )
    psth_example[1].hlines(
        np.mean(norm_results), 0, plot_lims[1], color=d1_color, ls="dotted", lw=1
    )
    psth_example[1].set_ylabel("Norm. Firing Rate")

    # Plot changes in FR and CV for pre-stim vs stim periods
    gs1CD = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs1[1], width_ratios=[1, 1], wspace=0.35
    )
    fr_cv_change = [fig.add_subplot(gs1CD[0]), fig.add_subplot(gs1CD[1])]

    category_shift_single(
        data_short,
        "baseline_freq",
        "stim_freq",
        fr_cv_change[0],
        tick1="Pre",
        tick2="Stim",
        color1=d1_color,
        ylabel="Hz",
        title="Firing Rate",
    )

    category_shift_single(
        data_short[data_short["neural_response"] != "complete inhibition"],
        "baseline_cv",
        "stim_cv",
        fr_cv_change[1],
        tick1="Pre",
        tick2="Stim",
        ylabel="CV",
        color1=d1_color,
        title="CV",
    )

    # Set up figure section for examples of different responses
    gs_class_examples = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=gs[2], hspace=0.05, wspace=0.4
    )
    examples = [
        fig.add_subplot(gs_class_examples[i, j]) for i in range(2) for j in range(4)
    ]

    # Plot each example
    plot_example(
        data_short, [examples[0], examples[4]], [-1, 1], 120, "complete inhibition"
    )
    plot_example(
        data_short,
        [examples[1], examples[5]],
        [-1, 1],
        135,
        "partial inhibition",
    )
    plot_example(
        data_short,
        [examples[2], examples[6]],
        [-1, 1],
        23,
        "adapting inhibition",  # 30, 62, 49
    )
    plot_example(
        data_short, [examples[3], examples[7]], [-1, 1], 64, "excitation"
    )  # 101

    # Set up figure section for classification results
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3], wspace=0.25)
    classifications = [fig.add_subplot(gs3[i]) for i in range(3)]

    # Plot distribution of responses
    classification_plot(data_short, classifications[0], plot_vals=True, abbrev=True)

    # Plot responses by location
    locations_plot_d1(data_short, classifications[1])

    # Plot responses by modulation factor and baseline firing rate
    baseline_class_prediction_MF(data_short, classifications[2])

    # Label figure and clean up
    add_fig_labels(
        [
            psth_example[0],
            fr_cv_change[0],
            examples[0],
            classifications[0],
            classifications[1],
            classifications[2],
        ],
        start_at=2,
    )

    # Save and display figure
    plt.savefig(f"{save_dir}/figure_1.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close()
    run_cmd(f"open {save_dir}/figure_1.pdf")
