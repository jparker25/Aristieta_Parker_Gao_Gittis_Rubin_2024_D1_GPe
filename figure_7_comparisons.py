"""
File: figure_7_comparisons.py
Author: John E. Parker
Date: 13 February 2024

Description: Generates figure 7 comparing d1 and gpe in naive and dd conditions.
"""

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib import *

# import user modules
from helpers import *
from generate_figures import *


# Generate figure based on all data, comparing d1 and gpe in naive and dd conditions.
def generate_figure(
    d1_naive_short, d1_dd_short, gpe_naive_short, gpe_dd_short, save_dir
):
    """
    Generates figure 7, comparing d1 with gpe in naive and dd conditions..

    d1_naive_short\t:\tdataframe of d1 naive data
    d1_dd_short\t:\tdataframe of d1 dd data
    gpe_naive_short\t:\tdataframe of gpe naive data
    gpe_dd_short\t:\tdataframe of gpe dd data
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
    fig, ax = plt.subplots(2, 1, figsize=(4, 3), dpi=300, tight_layout=True)
    axes = [ax[i] for i in range(2)]

    """
    fig = plt.figure(figsize=(6, 3), dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    """

    # plot normalized PSTH relative to 1s of pre-stim firing for naive data
    psth_norm_compare(
        d1_naive_short,
        gpe_naive_short,
        axes[0],
        bw=0.05,
        baseline=2,
        stim=1,
        post=0,
        plot_lims=[-1, 1],
        drawboth=True,
        color1=d1_color,
        semcolor1=d1_semcolor,
        color2=gpe_color,
        semcolor2=gpe_semcolor,
        label1=f"D1-ctl ($n={len(d1_naive_short)}$)",
        label2=f"GPe-ctl ($n={len(gpe_naive_short)}$)",
        lw=1,
        linestyle="solid",
    )

    """
    # Plot naive modulation factors for FR and compare
    category_change_MF(
        d1_naive_short,
        gpe_naive_short,
        "baseline_freq",
        "stim_freq",
        axes[1],
        color1=d1_color,
        color2=gpe_color,
        ylabel="MF",
        title="Modulation Factor",
        tick1="D1-ctl",
        tick2="GPe-ctl",
        ylims=[-1, 0.3],
    )
    """

    # plot normalized PSTH relative to 1s of pre-stim firing for dd data
    psth_norm_compare(
        d1_dd_short,
        gpe_dd_short,
        axes[1],  # axes[2],
        bw=0.05,
        baseline=2,
        stim=1,
        post=0,
        plot_lims=[-1, 1],
        drawboth=True,
        color1=d1_dd_color,
        semcolor1=d1_dd_semcolor,
        color2=gpe_dd_color,
        semcolor2=gpe_dd_semcolor,
        label1=f"D1-DD ($n={len(d1_dd_short)}$)",
        label2=f"GPe-DD ($n={len(gpe_dd_short)}$)",
        lw=1,
        linestyle="solid",
    )

    """
    # Plot dd modulation factors for FR and compare
    category_change_MF(
        d1_dd_short,
        gpe_dd_short,
        "baseline_freq",
        "stim_freq",
        axes[3],
        color1=d1_dd_color,
        color2=gpe_dd_color,
        ylabel="MF",
        title="Modulation Factor",
        tick1="D1-DD",
        tick2="GPe-DD",
        ylims=[-1, 0.3],
    )
    """

    # Label figure and clean up
    add_fig_labels(axes)
    # Save and display figure
    plt.savefig(f"{save_dir}/figure_7.pdf", bbox_inches="tight")
    plt.close()
    run_cmd(f"open {save_dir}/figure_7.pdf")
