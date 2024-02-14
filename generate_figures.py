"""
File: generate_figures.py
Author: John E. Parker
Date: 13 February 2024

Description: Creates panels of different figures.
"""

import numpy as np
import seaborn as sns
import pickle
from scipy import stats
from matplotlib import *
import matplotlib.patches as patches

# import user modules
from helpers import *
import analyze_data
import run_statistics

color_dict = {
    "complete inhibition": "navy",
    "adapting inhibition": "cornflowerblue",
    "partial inhibition": "blue",
    "no effect": "slategrey",
    "excitation": "lightcoral",
    "biphasic IE": "blueviolet",
    "biphasic EI": "orchid",
}
types = [
    "no effect",
    "complete inhibition",
    "partial inhibition",
    "adapting inhibition",
    "excitation",
]


# Generate raster plot of first trial from provided data set
def plot_random_trials(data, axes, plot_lims, n=10, baseline=1, stim=1, post=0):
    """
    Generates raster plot of first trail from provided data set.

    data\t:\tdataframe to sample from
    axes\t:\taxes to plot rasters on
    plot_lims\t:\thoritzontal plot limits
    n\t:\tnumber of neurons to sample from (default = 10)
    baseline\t:\tlength of time for baseline (default = 1)
    stim\t:\tnlength of time for stimulus (default = 1)
    post\t:\tlength of time for pist (default = 0)
    """
    # Set seed for reproducibility control
    np.random.seed(24)

    # Sample from n neurons in data
    sample = data.sample(n=n)

    # iterate through all n neurons and gather spikes of each trial 1
    all_spikes = []
    for _, row in sample.iterrows():
        neuron = pickle.load(open(f"{row['cell_dir']}/neuron.obj", "rb"))
        light_on = np.loadtxt(f"{row['cell_dir']}/light_on.txt")
        spikes = (
            neuron.spikes[
                (neuron.spikes >= light_on[0] - baseline)
                & (neuron.spikes < light_on[0] + stim + post)
            ]
            - light_on[0]
        )
        all_spikes.append(spikes)

    # Plot as raster
    axes.eventplot(all_spikes, colors="k", lw=0.25)

    # clean up figure
    axes.set_ylim([-0.5, n - 0.5])
    axes.spines[["left", "bottom", "right", "top"]].set_visible(False)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_ylabel("Cell Number")
    axes.set_xlim(plot_lims)
    ylim = axes.get_ylim()

    # Add highlight for stimulus period
    rect = patches.Rectangle(
        (0, ylim[0]),
        stim,
        ylim[1] - ylim[0],
        color="#0046ff",
        alpha=0.05,
        edgecolor=None,
    )
    axes.add_patch(rect)
    axes.set_xlim([-1, 1.1])


def baseline_class_prediction_MF(data, axes):
    """
    Generates MF vs baseline firing rate plot, color coded by response.

    data\t:\tdataframe for baseline FR and MF
    axes\t:\taxes to plot rasters on
    """

    # Generate MF data
    all_rates, inh_mfs, ex_mfs, ne_mfs = analyze_data.get_modulation_factors(
        data, "frequency"
    )

    # Iterate through types and plot MF vs baseline FR
    for k in range(len(types)):
        sns.scatterplot(
            x=all_rates[all_rates[:, 2] == k, 1],
            y=all_rates[all_rates[:, 2] == k, 0],
            color=color_dict[types[k]],
            ax=axes,
            label=f"{types[k].title()}",
            s=15,
        )

    # Collect MFs by inhibition, excitation, and no  effect
    inh_mfs = np.asarray(inh_mfs)
    ex_mfs = np.asarray(ex_mfs)
    ne_mfs = np.asarray(ne_mfs)

    # Remove NANs
    all_rates = all_rates[~np.isnan(all_rates).any(axis=1)]

    # Calculate and plot linear regression
    m, b, rval, pval, stderr = stats.linregress(all_rates[:, 1], all_rates[:, 0])
    xlims = np.asarray([-1.1, 1.1])
    axes.plot(xlims, m * xlims + b, ls="dashed", color="k", lw=1)

    # Plot data and clean figure
    axes.vlines(0, 0, 75, color="k", ls="dashed", lw=0.5)
    axes.set_ylim([0, axes.get_ylim()[1] + 20])
    axes.annotate(
        f"$r=${rval:.02f}",
        xycoords="data",
        xy=(0.85, m * 0.85 + b + 3),
        color="k",
        fontsize=6,
        ha="center",
    )
    axes.legend(fancybox=False, frameon=False, fontsize=6)
    axes.set_ylim([0, 120])
    axes.set_xlim(xlims)
    axes.set_xlabel("Modulation Factor")
    axes.set_ylabel("Baseline FR (Hz)")
    sns.despine(ax=axes)


def plot_example(
    data, axes, plot_lims, cell_num, example_type, bw=0.05, baseline=1, stim=1, post=0
):
    """
    Plot one example of a certain response.

    data\t:\tdataframe to sample from
    axes\t:\tArray of two sub plots
    plot_lims\t:\thoritzontal plot limits
    cell_num\t:\tcell number to grab example
    example_type\t:\tresponse title
    bw\t:\tbin width for PSTH (default = 0.05)
    baseline\t:\tlength of time for baseline (default = 1)
    stim\t:\tnlength of time for stimulus (default = 1)
    post\t:\tlength of time for pist (default = 0)
    """

    # Generate bins for PSTH
    bins = np.arange(-baseline, stim + post + bw, bw)

    # Grab example from cell_number and data set
    plot_example = data[data["cell_num"] == cell_num]
    src = plot_example.iloc[0]["cell_dir"]
    neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
    light_on = np.loadtxt(f"{src}/light_on.txt")

    # Iterate through trials and grab all spikes and PSTH values
    neuron_trials = []
    total_fr = np.zeros(len(bins) - 1)
    all_spikes = []

    for trial in range(neuron.trials):
        spikes = (
            neuron.spikes[
                (neuron.spikes >= light_on[trial] - baseline)
                & (neuron.spikes < light_on[trial] + stim + post)
            ]
            - light_on[trial]
        )
        fr, _ = np.histogram(spikes, bins=bins)
        total_fr += (fr / bw) / neuron.trials
        neuron_trials.append(fr)
        all_spikes.append(spikes)

    # Plot raster of trials
    axes[0].eventplot(all_spikes, colors="k", lw=0.25)
    axes[0].spines[["left", "bottom", "right", "top"]].set_visible(False)
    axes[0].set_ylim([-0.5, neuron.trials - 0.5])
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_ylabel("Trial #")

    # Plot PSTH
    axes[1].bar(bins[:-1], total_fr, color="k", align="edge", width=bw * 0.9)
    axes[1].set_xlabel("Time (s)", fontsize=8)
    axes[1].set_ylabel("FR (Hz)", fontsize=8)
    axes[1].set_ylim([0, np.max(total_fr)])
    axes[0].set_title(example_type.title(), fontsize=8)

    # Color stimulus period and clean up figure
    for i in [0, 1]:
        axes[i].set_xlim(plot_lims)
        ylim = axes[i].get_ylim()
        rect = patches.Rectangle(
            (0, ylim[0]),
            stim,
            ylim[1] - ylim[0],
            color="#0046ff",
            alpha=0.05,
            edgecolor=None,
        )
        axes[i].add_patch(rect)
        sns.despine(ax=axes[1])
    match_axis(axes, type="x")


def classification_plot(data, axes, plot_vals=False, abbrev=False):
    """
    Plot distribution of responses.

    data\t:\tdataframe to sample from
    axes\t:\taxes to plot on
    plot_vals\t:\t if true, plot amount in category above bar (default = False)
    abbrev\t:\t if true abbreviate x-ticks (default = False)
    """

    # Count number of cells in each response type and set as percentage
    classes = np.zeros(5)
    for type in range(len(classes)):
        classes[type] = data[(data["neural_response"] == types[type])].shape[0]
    counts = [int(classes[x]) for x in range(len(classes))]
    classes = (classes / np.sum(classes)) * 100

    # Plot values as bar graph
    pos = 1
    step = 1
    width = 0.9
    for i in range(5):
        axes.bar(
            pos,
            classes[i],
            color=color_dict[types[i]],
            edgecolor="k",
            linewidth=1,
            width=width,
            align="edge",
        )

        # Annotate bars
        if plot_vals:
            axes.annotate(
                f"$n={counts[i]}$",
                xycoords="data",
                xy=(pos + width / 2, classes[i] + 2),
                color=color_dict[types[i]],
                fontsize=6,
                ha="center",
            )
        pos += step

    # Clean up plot
    axes.set_xticks(np.arange(1 + width / 2, pos, step))
    xticklabels = [
        "No Effect",
        "Complete\nInhibition",
        "Partial\nInhibition",
        "Adapting\nInhibition",
        "Excitation",
    ]

    # Abbreviate xticks
    if abbrev:
        xticklabels = ["NE", "CI", "PI", "AI", "EX"]
    axes.set_xticklabels(
        xticklabels,
        fontsize=8,
    )

    axes.set_ylabel("Percentage")
    axes.set_xlim([width, pos])
    sns.despine(ax=axes)


def classification_plot_comparison_horizontal(
    data, data2, axes, text1, text2, plot_vals=True, abbrev=True
):
    """
    Plot two distributions of responses.

    data\t:\tdataframe to sample from
    data2\t:\tdataframe to compare with
    axes\t:\taxes to plot on
    text1\t:\tlabel for data set 1
    text2\t:\tlabel for data set 2
    plot_vals\t:\t if true, plot amount in category above bar (default = True)
    abbrev\t:\t if true abbreviate x-ticks (default = True)
    """

    # Set yticklabels and abbreviate if necessary
    yticklabels = [
        "No Effect",
        "Complete\nInhibition",
        "Partial\nInhibition",
        "Adapting\nInhibition",
        "Excitation",
    ]
    if abbrev:
        yticklabels = ["NE", "CI", "PI", "AI", "EX"]

    # Count number of cells in each response type and set as percentage
    classes = np.zeros(5)
    for type in range(len(classes)):
        classes[type] = data[(data["neural_response"] == types[type])].shape[0]
    counts = [int(classes[x]) for x in range(len(classes))]
    classes = (classes / np.sum(classes)) * 100

    # Count number of cells in each response type and set as percentage
    classes2 = np.zeros(5)
    for type in range(len(classes2)):
        classes2[type] = data2[(data2["neural_response"] == types[type])].shape[0]
    counts2 = [int(classes2[x]) for x in range(len(classes2))]
    classes2 = (classes2 / np.sum(classes2)) * 100

    # Plot values as bar graph
    pos = 1
    step = 1
    width = 0.9
    for i in range(5):
        axes.barh(
            pos,
            classes[i],
            color=color_dict[types[i]],
            edgecolor="k",
            linewidth=1,
            height=width,
            align="edge",
        )
        if plot_vals:
            axes.annotate(
                # f"{classes[i]:.02f}%",
                f"$n={counts[i]}$",
                xycoords="data",
                xy=(
                    classes[i] + 5,
                    pos + width / 2,
                ),
                color=color_dict[types[i]],
                fontsize=6,
                ha="left",
            )
        pos += step

    # Plot values as bar graph
    pos = 1
    step = 1
    for i in range(5):
        axes.barh(
            pos,
            -classes2[i],
            color=color_dict[types[i]],
            edgecolor="k",
            linewidth=1,
            height=width,
            align="edge",
            alpha=0.5,
        )
        if plot_vals:
            axes.annotate(
                f"$n={counts2[i]}$",
                xycoords="data",
                xy=(
                    -classes2[i] - 5,
                    pos + width / 2,
                ),
                color=color_dict[types[i]],
                fontsize=6,
                ha="right",
            )
        pos += step

    # Clean up y-ticks
    axes.set_yticks(np.arange(1 + width / 2, pos, step))
    axes.set_yticklabels(yticklabels, fontsize=8)
    axes.set_xlabel("Percentage")

    # Clean up and annotate figure
    ylims = axes.get_ylim()
    axes.vlines(0, ylims[0], ylims[1], ls="dashed", lw=0.5, color="k")
    axes.set_ylim([ylims[0], axes.get_ylim()[1]])
    axes.annotate(
        text1,
        xycoords="data",
        xy=(40, axes.get_ylim()[1] * 0.95),
        fontsize=8,
        zorder=15,
        ha="center",
    )
    axes.annotate(
        text2,
        xycoords="data",
        xy=(-40, axes.get_ylim()[1] * 0.95),
        fontsize=8,
        zorder=15,
        ha="center",
    )
    axes.set_xticks(np.arange(-80, 80 + 20, 20))
    axes.set_xticklabels(np.abs(np.arange(-80, 80 + 20, 20)))
    sns.despine(ax=axes)


def psth_norm(
    data,
    axes,
    bw=0.05,
    baseline=2,
    stim=1,
    post=0,
    plot_lims=[-1, 1],
    color="k",
    semcolor="lightgray",
    draw_rect=True,
    plot_sem=True,
    label="Naive D1",
    zorder=10,
    alpha=0.25,
    lw=1,
    linestyle="solid",
):
    """
    Plots normalized PSTH and returns average normalized stimulus firing rate.

    data\t:\tdataframe to sample from
    axes\t:\taxes to plot on
    bw\t:\tbin width for PSTH (default = 0.05)
    baseline\t:\tbaseline length to plot (default = 2)
    stim\t:\tstimulus length to plot (default = 1)
    post\t:post length to plot (default = 0)
    plot_lims\t:\tSet x-lims for plotting (default = [-1,1])
    color\t:\tcolor to set mean PSTH (default = "k")
    semcolor\t:\tcolor to set SEM around mean (default = "lightgray")
    draw_rect\t:\tdraw stimulus bar if true (default = True)
    plot_sem\t:\tplot SEM if true (default = True)
    label\t:\tlabel for legend (default = "Naive D1")
    zorder\t:\twhere to place aspects of plot for visibility (default = 10)
    alpha\t:\ttransparency level of highlighted stim (default = 0.25)
    lw\t:\tlinewidth for PSTH mean line (default = 1)
    linestyle\t:\tpattern of PSTH mean line (default = "solid")
    """

    # Set up bins
    bins = np.arange(-baseline, stim + post + bw, bw)

    # Gather normalized FRs
    all_average, average_stim_norm_fr = analyze_data.get_norm_frs(
        data, baseline=baseline, stim=stim, post=post, bw=bw
    )

    # Plot mean PSTH and SEM
    all_average = np.asarray(all_average)
    if all_average.shape[0] > 0:
        mean_all_average = np.mean(all_average, axis=0)
        sem = stats.sem(all_average, axis=0)

        axes.plot(
            bins[1:],
            mean_all_average,
            color=color,
            linewidth=lw,
            label=f"{label}",
            zorder=zorder,
            ls=linestyle,
        )
        axes.set_ylim([0, 1.2])
        axes.legend(loc="lower left", frameon=False, fancybox=False, fontsize="small")
        if plot_sem:
            axes.fill_between(
                bins[1:],
                mean_all_average + sem,
                mean_all_average - sem,
                color=semcolor,
                edgecolor=semcolor,
                zorder=zorder - 1,
                alpha=alpha,
            )

        # Clean up plot
        axes.set_xlim(plot_lims)
        axes.set_xlabel("Time (s)")
        if draw_rect:
            ylim = axes.get_ylim()
            rect = patches.Rectangle(
                (0, ylim[1] * 0.95), stim, ylim[1] * 0.05, color="#0046ff"
            )
            axes.add_patch(rect)
            rect = patches.Rectangle(
                (0, ylim[0]),
                stim,
                ylim[1] - ylim[0],
                color="#0046ff",
                alpha=0.05,
                edgecolor=None,
            )
            axes.add_patch(rect)
        sns.despine(ax=axes)
    # Set lims
    axes.set_xlim([-1, 1.1])
    return average_stim_norm_fr


def psth_norm_compare(
    data1,
    data2,
    axes,
    bw=0.05,
    baseline=2,
    stim=1,
    post=0,
    plot_lims=[-1, 1],
    label1="D1 MSNs",
    label2="GPe-PV",
    color1="k",
    semcolor1="lightgray",
    color2="blue",
    semcolor2="lightblue",
    drawboth=False,
    lw=1,
    linestyle="dashed",
):
    """
    Plots normalized PSTH and returns average normalized stimulus firing rate.

    data1\t:\tdataframe to sample from
    data2\t:\tdataframe to sample from
    axes\t:\taxes to plot on
    bw\t:\tbin width for PSTH (default = 0.05)
    baseline\t:\tbaseline length to plot (default = 2)
    stim\t:\tstimulus length to plot (default = 1)
    post\t:post length to plot (default = 0)
    plot_lims\t:\tSet x-lims for plotting (default = [-1,1])
    label1\t:\tlabel for first data set (default = "D1 MSNs")
    label2\t:\tlabel for first data set (default = "GPe-PV")
    color1\t:\tcolor to set mean PSTH for data1 (default = "k")
    semcolor1\t:\tcolor to set SEM around mean for data1 (default = "lightgray")
    color2\t:\tcolor to set mean PSTH for data2 (default = "blue")
    semcolor2\t:\tcolor to set SEM around mean for data2 (default = "lightblue")
    drawboth\t:\tif true draw both SEM (default = False)
    lw\t:\tlinewidth for PSTH mean line (default = 1)
    linestyle\t:\tpattern of PSTH mean line (default = "dashed")
    """

    # Plot normalized PSTH for data1 and return average stim norm firing rate
    norm_fr1 = psth_norm(
        data1,
        axes=axes,
        bw=bw,
        baseline=baseline,
        stim=stim,
        post=post,
        plot_lims=plot_lims,
        color=color1,
        semcolor=semcolor1,
        draw_rect=False,
        label=label1,
        zorder=10,
        alpha=0.25,
        lw=lw,
    )

    # Plot normalized PSTH for data2 and return average stim norm firing rate
    norm_fr2 = psth_norm(
        data2,
        axes=axes,
        bw=bw,
        baseline=baseline,
        stim=stim,
        post=post,
        plot_lims=plot_lims,
        color=color2,
        semcolor=semcolor2,
        draw_rect=True,
        plot_sem=True if drawboth else False,
        label=label2,
        zorder=9,
        alpha=0.25,
        lw=lw,
        linestyle=linestyle,
    )

    # Remove nans and plot mean of data1 and data2
    x1 = np.asarray(norm_fr1)
    x2 = np.asarray(norm_fr2)
    x1 = x1[np.logical_not(np.isnan(x1))]
    x2 = x2[np.logical_not(np.isnan(x2))]
    axes.hlines(np.mean(x1), 0, plot_lims[1], color=color1, ls="dotted", lw=1)
    axes.hlines(np.mean(x2), 0, plot_lims[1], color=color2, ls="dotted", lw=1)

    # Run unpaired t-test and plot stats
    _, pval = stats.ttest_ind(
        x1, x2, alternative="less" if np.mean(x1) < np.mean(x2) else "greater"
    )
    plot_vertical_bracket(
        axes,
        1.01,
        1.04,
        np.mean(x1),
        np.mean(x2),
        run_statistics.pval_string(pval),
        color="k",
        lw=1,
    )

    # Label figure
    axes.set_ylabel("Norm. Firing Rate", fontsize=8)
    axes.set_xlabel("Time (s)", fontsize=8)
    axes.set_xlim([-1, 1.1])


def category_shift_single(
    data1,
    category1,
    category2,
    axes,
    color1="k",
    tick1="Pre",
    tick2="Stim",
    title="Comparison",
    ylabel="Firing Rate (Hz)",
):
    """
    Plots pre-stim and stim values of data set.

    data1\t:\tdataframe to sample from
    category1\t:\tpre-stim data value
    category2\t:\tstim data value
    axes\t:\taxes to plot on
    color1\t:\tcolor to set mean for data1 (default = "k")
    tick1\t:\tlabel for tick1 (default = "Pre")
    tick2\t:\tlabel for tick2 (default = "Stim")
    title\t:\ttitle for plot (default = "Comparison")
    ylabel\t:\tlabel for y-axis (default = "Firing Rate (Hz)")
    """

    # Plot average values of category 1 and category 2 for data1
    axes.scatter(
        [0, 1],
        [np.mean(data1[category1]), np.mean(data1[category2])],
        color=color1,
        zorder=10,
        s=20,
    )
    axes.plot(
        [0, 1],
        [np.mean(data1[category1]), np.mean(data1[category2])],
        color=color1,
        zorder=10,
    )

    # Plot errorbars of category 1 and category 2 for data1
    axes.errorbar(
        0,
        np.mean(data1[category1]),
        yerr=np.std(data1[category1]),
        color=color1,
        zorder=10,
        capsize=2,
    )
    axes.errorbar(
        1,
        np.mean(data1[category2]),
        yerr=np.std(data1[category2]),
        color=color1,
        zorder=10,
        capsize=2,
    )

    # Plot all individual category1 to category2 shifts for cells in data1
    for _, row in data1.iterrows():
        axes.plot(
            [0, 1],
            [row[category1], row[category2]],
            marker="o",
            color="gray",
            markersize=0.5,
            lw=0.25,
            zorder=5,
            alpha=0.25,
        )

    # Run paired t-test and plot stat result
    _, pval1 = stats.ttest_rel(data1[category1], data1[category2])
    ylim1 = axes.get_ylim()[1]
    plot_bracket(
        axes,
        0,
        1,
        ylim1 * 1,
        ylim1 * 1.05,
        run_statistics.pval_string(pval1, paired=True),
        lw=1,
    )

    # Clean up and label figure
    axes.set_xticks([0, 1])
    axes.set_xticklabels([tick1, tick2], fontsize=8)
    axes.set_xlim([-0.25, 1.25])
    axes.set_title(title, fontsize=8)
    axes.set_ylabel(ylabel, fontsize=8)
    sns.despine(ax=axes)


def category_change_MF(
    data1,
    data2,
    category1,
    category2,
    axes,
    color1="k",
    color2="k",
    tick1="D1-MSNs",
    tick2="GPe-PV",
    ylabel="Modulation Factor",
    title="All Responses",
    ylims=[-0.5, 0.5],
):
    """
    Plots average MF and errorbars for data1 and data2.

    data1\t:\tdataframe to sample from
    data2\t:\tdataframe to compare with
    category1\t:\tcategory for MF calculation
    category2\t:\tcategory for MF calculation
    axes\t:\taxes to plot on
    color1\t:\tcolor to set bar MF for data1 (default = "k")
    color2\t:\tcolor to set bar MF for data2 (default = "k")
    tick1\t:\tlabel for tick1 (default = "D1-MSNs")
    tick2\t:\tlabel for tick2 (default = "GPe-PV")
    ylabel\t:\tlabel for y-axis (default = "Modulation Factor")
    title\t:\ttitle for plot (default = "All Responses")
    ylims\t:\tset limits for y-axis (default = [-0.5,0.5])
    """

    # Calculate all modulation factors for data set 1
    y1 = np.zeros(len(data1))
    count = 0
    for _, row in data1.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
        modulation_factors = analyze_data.calc_mf(neuron, category1)
        if len(modulation_factors) > 0:
            y1[count] = np.mean(modulation_factors)
        count += 1

    # Calculate all modulation factors for data set 2
    y2 = np.zeros(len(data2))
    count = 0
    for _, row in data2.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
        modulation_factors = analyze_data.calc_mf(neuron, category1)
        if len(modulation_factors) > 0:
            y2[count] = np.mean(modulation_factors)
        count += 1

    # Remove NANs
    y1 = y1[~np.isnan(y1)]
    y2 = y2[~np.isnan(y2)]

    # Plot MFs with errorbars
    a1 = 0.1
    b1 = 0.15
    a2 = 0.18
    b2 = a2 + (b1 - a1)
    x1 = np.random.rand(len(y1)) * (b1 - a1) + a1
    x2 = np.random.rand(len(y2)) * (b2 - a2) + a2
    axes.bar(
        np.mean(x1),
        np.mean(y1),
        width=0.07,
        color=color1,
        zorder=10,
    )
    plot, caps, bars = axes.errorbar(
        np.mean(x1),
        np.mean(y1),
        yerr=np.std(y1),
        lolims=True if np.mean(y1) > 0 else False,
        uplims=True if np.mean(y1) < 0 else False,
        color=color1,
        zorder=10,
        capsize=3,
        fmt="",
    )
    caps[0].set_marker("_")

    axes.bar(
        np.mean(x2),
        np.mean(y2),
        width=0.07,
        color=color2,
        zorder=10,
    )

    plot, caps, bars = axes.errorbar(
        np.mean(x2),
        np.mean(y2),
        yerr=np.std(y2),
        lolims=True if np.mean(y2) > 0 else False,
        uplims=True if np.mean(y2) < 0 else False,
        color=color2,
        zorder=10,
        capsize=3,
        fmt="",
        markersize=0,
    )
    caps[0].set_marker("_")

    # Run unpaired t-test and plot stats
    _, pval = stats.ttest_ind(
        y1, y2, alternative="less" if np.mean(y1) < np.mean(y2) else "greater"
    )
    ybracket1 = np.abs(ylims[1] - ylims[0]) * 0.05
    ybracket2 = np.abs(ylims[1] - ylims[0]) * 0.1

    plot_bracket(
        axes,
        (a1 + b1) / 2,
        (a2 + b2) / 2,
        ybracket1,
        ybracket2,
        run_statistics.pval_string(pval),
        lw=1,
    )

    # Clean up figure
    axes.set_ylim(ylims)
    axes.set_xticks([np.mean(x1), np.mean(x2)])
    axes.set_xticklabels([tick1, tick2], fontsize=8)
    axes.set_ylabel(ylabel, fontsize=8)
    xlims = axes.get_xlim()
    axes.hlines(0, xlims[0], xlims[1], color="gray", ls="dashed", lw=0.5)
    axes.set_title(title, fontsize=8)
    sns.despine(ax=axes)


def locations_plot_d1(data, axes):
    """
    Plots bars of response distribution for d1 data within medial, central, and lateral locations.

    data\t:\tdataframe to sample from
    axes\t:\taxes to plot on
    """

    # Grab location data and set to percents
    locs = analyze_data.get_d1_locations(data)
    counts = [np.sum(locs[i, :], dtype=int) for i in range(3)]

    # Plot data by axes
    width = 0.9
    for i in range(3):
        locs[i, :] = locs[i, :] * 100 / np.sum(locs[i, :])
        for k in range(5):
            axes.bar(
                i,
                locs[i, k],
                bottom=np.sum(locs[i, :k]),
                color=color_dict[types[k]],
                align="edge",
                width=width,
                edgecolor="k",
            )

    # Clean up figure and label appropriately
    axes.set_xticks([width / 2, 1 + width / 2, 2 + width / 2])
    axes.set_xticklabels(
        [
            f"Medial\n$n={counts[0]}$",
            f"Central\n$n={counts[1]}$",
            f"Lateral\n$n={counts[2]}$",
        ],
        fontsize=8,
    )
    axes.set_ylabel("Percentage")
    sns.despine(ax=axes)


def locations_plot_gpe(data, axes):
    """
    Plots bars of response distribution of gpe data within medial, central, and lateral locations.

    data\t:\tdataframe to sample from
    axes\t:\taxes to plot on
    """

    # Grab location data and set to percents
    locs = analyze_data.get_gpe_locations(data)
    counts = [np.sum(locs[i, :], dtype=int) for i in range(3)]

    # Plot data by axes
    width = 0.9
    for i in range(3):
        locs[i, :] = locs[i, :] * 100 / np.sum(locs[i, :])
        for k in range(5):
            axes.bar(
                i,
                locs[i, k],
                bottom=np.sum(locs[i, :k]),
                color=color_dict[types[k]],
                align="edge",
                width=width,
                edgecolor="k",
            )

    # Clean up figure and label appropriately
    axes.set_xticks([width / 2, 1 + width / 2, 2 + width / 2])
    axes.set_xticklabels(
        [
            f"Medial\n$n={counts[0]}$",
            f"Central\n$n={counts[1]}$",
            f"Lateral\n$n={counts[2]}$",
        ],
        fontsize=8,
    )
    axes.set_ylabel("Percentage")
    sns.despine(ax=axes)


def plot_bracket(ax, x1, x2, y1, y2, text, color="k", lw=1):
    """
    Plots horizontal bracket with statistical result.

    ax\t:\taxes to plot on
    x1\t:\tlower x-bound to start bracket
    x2\t:\tupper x-bound to end bracket
    y1\t:\tlower y-bound to start bracket
    y2\t:\tupper y-bound to end bracket
    text\t:\tlabel for statistical reesult
    color\t:\tcolor of text (default = "k")
    lw\t:\tlinewidth of bracket (default = 1)
    """

    # Plot horizontal lines of bracket
    ax.hlines(y2, x1, x2, color=color, lw=lw, zorder=15)

    # Plot vertical lines of bracket
    ax.vlines([x1, x2], y1, y2, color=color, lw=lw, zorder=15)

    # Annotate bracket
    ax.annotate(
        text,
        xycoords="data",
        xy=((x1 + x2) / 2, y2 + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01),
        fontsize=12 if text != "n.s." else 8,
        zorder=15,
        ha="center",
    )


def plot_vertical_bracket(ax, x1, x2, y1, y2, text, color="k", lw=1):
    """
    Plots vertical bracket with statistical result.

    ax\t:\taxes to plot on
    x1\t:\tlower x-bound to start bracket
    x2\t:\tupper x-bound to end bracket
    y1\t:\tlower y-bound to start bracket
    y2\t:\tupper y-bound to end bracket
    text\t:\tlabel for statistical reesult
    color\t:\tcolor of text (default = "k")
    lw\t:\tlinewidth of bracket (default = 1)
    """
    # Plot horizontal lines of bracket
    ax.hlines([y1, y2], x1, x2, color=color, lw=lw, zorder=15)

    # Plot vertical lines of bracket
    ax.vlines(x2, y1, y2, color=color, lw=lw, zorder=15)

    # Annotate bracket
    ax.annotate(
        text,
        xycoords="data",
        xy=(x2 + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01, (y1 + y2) / 2),
        fontsize=12 if text != "n.s." else 8,
        zorder=15,
        va="center",
    )
