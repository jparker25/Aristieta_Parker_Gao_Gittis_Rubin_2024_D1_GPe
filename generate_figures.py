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


# Generate raster plot of first trial from provided data set
def plot_random_trials(data, axes, plot_lims, n=10, baseline=1, stim=1, post=0):
    """
    Generates raster plot of first trail from provided data set.

    data\t:\tdataframe to sample from
    axes\t:\taxes to plot rasters on
    plot+lims\t:\thoritzontal plot limits
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
    all_rates = np.zeros((len(data), 2))
    all_count = 0
    ne_mfs = []
    inh_mfs = []
    ex_mfs = []
    for response in types:
        subset = data[data["neural_response"] == response]
        rates = np.zeros((len(subset), 2))
        count = 0
        for _, row in subset.iterrows():
            src = row["cell_dir"]
            neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
            # neuron_rates = np.zeros((neuron.trials, 2))
            modulation_factors = []
            for trial in range(neuron.trials):
                stim_rate = pickle.load(
                    open(
                        f"{src}/trial_{trial+1:02d}/stimulus_data/stimulus_spike_train.obj",
                        "rb",
                    )
                ).freq

                base_rate = pickle.load(
                    open(
                        f"{src}/trial_{trial+1:02d}/baseline_data/baseline_spike_train.obj",
                        "rb",
                    )
                ).freq
                if (stim_rate + base_rate) > 0:
                    modulation_factors.append(
                        (stim_rate - base_rate) / (stim_rate + base_rate)
                    )
            if len(modulation_factors) > 0:
                rates[count, 0] = row["pre_exp_freq"]
                rates[count, 1] = np.mean(modulation_factors)

            if "inhibition" in response:
                inh_mfs.append(rates[count, 1])
            elif "excitation" in response:
                ex_mfs.append(rates[count, 1])
            else:
                ne_mfs.append(rates[count, 1])
            all_rates[all_count, :] = [rates[count, 0], rates[count, 1]]
            count += 1
            all_count += 1
        sns.scatterplot(
            x=rates[:, 1],
            y=rates[:, 0],
            color=color_dict[response],
            ax=axes,
            label=f"{response.title()}",
            s=15,
        )

    inh_mfs = np.asarray(inh_mfs)
    ex_mfs = np.asarray(ex_mfs)
    ne_mfs = np.asarray(ne_mfs)

    all_rates = all_rates[~np.isnan(all_rates).any(axis=1)]
    m, b, rval, pval, stderr = stats.linregress(all_rates[:, 1], all_rates[:, 0])
    xlims = np.asarray([-1.1, 1.1])
    axes.plot(xlims, m * xlims + b, ls="dashed", color="k", lw=1)
    print(rval, pval)

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
    plot_example = data[data["cell_num"] == cell_num]
    src = plot_example.iloc[0]["cell_dir"]
    bins = np.arange(-baseline, stim + post + bw, bw)
    neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
    light_on = np.loadtxt(f"{src}/light_on.txt")
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
        # axes[0].scatter(spikes,np.ones(spikes.shape)*trial,marker=".",s=4,color="k")
    axes[0].eventplot(all_spikes, colors="k", lw=0.25)
    axes[0].spines[["left", "bottom", "right", "top"]].set_visible(False)
    axes[0].set_ylim([-0.5, neuron.trials - 0.5])
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_ylabel("Trial #")

    axes[1].bar(bins[:-1], total_fr, color="k", align="edge", width=bw * 0.9)
    axes[1].set_xlabel("Time (s)", fontsize=8)
    axes[1].set_ylabel("FR (Hz)", fontsize=8)
    axes[1].set_ylim([0, np.max(total_fr)])
    axes[0].set_title(example_type.title(), fontsize=8)
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
        "biphasic IE",
        "biphasic EI",
    ]

    classes = np.zeros(5)
    for type in range(len(classes)):
        classes[type] = data[(data["neural_response"] == types[type])].shape[0]
    counts = [int(classes[x]) for x in range(len(classes))]
    classes = (classes / np.sum(classes)) * 100
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
        if plot_vals:
            axes.annotate(
                # f"{classes[i]:.02f}%",
                f"$n={counts[i]}$",
                xycoords="data",
                xy=(pos + width / 2, classes[i] + 2),
                color=color_dict[types[i]],
                fontsize=6,
                ha="center",
            )
        pos += step
    axes.set_xticks(np.arange(1 + width / 2, pos, step))
    xticklabels = [
        "No Effect",
        "Complete\nInhibition",
        "Partial\nInhibition",
        "Adapting\nInhibition",
        "Excitation",
    ]
    if abbrev:
        xticklabels = ["NE", "CI", "PI", "AI", "EX"]
    axes.set_xticklabels(
        xticklabels,
        fontsize=8,
    )
    axes.set_ylabel("Percentage")
    axes.legend(frameon=False, fontsize="small")
    axes.set_xlim([width, pos])
    sns.despine(ax=axes)


def classification_plot_comparison_horizontal(
    data, data2, axes, text1, text2, plot_vals=True, abbrev=True
):
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
        "biphasic IE",
        "biphasic EI",
    ]

    yticklabels = [
        "No Effect",
        "Complete\nInhibition",
        "Partial\nInhibition",
        "Adapting\nInhibition",
        "Excitation",
    ]
    if abbrev:
        yticklabels = ["NE", "CI", "PI", "AI", "EX"]

    classes = np.zeros(5)
    for type in range(len(classes)):
        classes[type] = data[(data["neural_response"] == types[type])].shape[0]
    counts = [int(classes[x]) for x in range(len(classes))]
    classes = (classes / np.sum(classes)) * 100

    classes2 = np.zeros(5)
    for type in range(len(classes2)):
        classes2[type] = data2[(data2["neural_response"] == types[type])].shape[0]
    counts2 = [int(classes2[x]) for x in range(len(classes2))]
    classes2 = (classes2 / np.sum(classes2)) * 100

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

    axes.set_yticks(np.arange(1 + width / 2, pos, step))
    axes.set_yticklabels(yticklabels, fontsize=8)
    axes.set_xlabel("Percentage")

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


def classification_plot_comparison(data, data2, axes, plot_vals=True, abbrev=True):
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
        "biphasic IE",
        "biphasic EI",
    ]

    xticklabels = [
        "No Effect",
        "Complete\nInhibition",
        "Partial\nInhibition",
        "Adapting\nInhibition",
        "Excitation",
    ]
    if abbrev:
        xticklabels = ["NE", "CI", "PI", "AI", "EX"]

    classes = np.zeros(5)
    for type in range(len(classes)):
        classes[type] = data[(data["neural_response"] == types[type])].shape[0]
    counts = [int(classes[x]) for x in range(len(classes))]
    classes = (classes / np.sum(classes)) * 100

    classes2 = np.zeros(5)
    for type in range(len(classes2)):
        classes2[type] = data2[(data2["neural_response"] == types[type])].shape[0]
    classes2 = (classes2 / np.sum(classes2)) * 100

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
        if plot_vals:
            axes.annotate(
                # f"{classes[i]:.02f}%",
                f"$n={counts[i]}$",
                xycoords="data",
                xy=(
                    pos + width / 2,
                    classes[i] + 2 if classes[i] > classes2[i] else classes2[i] + 2,
                ),
                color=color_dict[types[i]],
                fontsize=6,
                ha="center",
            )
        pos += step

    pos = 1
    step = 1
    for i in range(5):
        axes.hlines(
            classes2[i], pos + 0.1, pos + step - 0.2, ls="dashed", lw=0.5, color="k"
        )
        axes.scatter(pos + 0.1, classes2[i], marker=">", color="k", s=2)
        axes.scatter(pos + step - 0.2, classes2[i], marker="<", color="k", s=2)
        pos += step

    axes.set_xticks(np.arange(1 + width / 2, pos, step))
    axes.set_xticklabels(xticklabels, fontsize=8)
    axes.set_ylabel("Percentage")
    axes.legend(frameon=False, fontsize="small")
    axes.set_xlim([width, pos])
    sns.despine(ax=axes)


def psth_norm(
    data,
    axes,
    bw=0.05,
    baseline=2,
    stim=10,
    post=2,
    plot_lims=[-2, 12],
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
    bins = np.arange(-baseline, stim + post + bw, bw)
    """
    all_average = []
    average_stim_norm_fr = []
    for _, row in data.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
        light_on = np.loadtxt(f"{src}/light_on.txt")
        neuron_trials = []
        neuron_stim = []
        for trial in range(neuron.trials):
            spikes = (
                neuron.spikes[
                    (neuron.spikes >= light_on[trial] - baseline)
                    & (neuron.spikes < light_on[trial] + stim + post)
                ]
                - light_on[trial]
            )
            stim_freq = pickle.load(
                open(
                    f"{row['cell_dir']}/trial_{trial+1:02d}/stimulus_data/stimulus_spike_train.obj",
                    "rb",
                )
            ).freq
            bl_freq = pickle.load(
                open(
                    f"{row['cell_dir']}/trial_{trial+1:02d}/baseline_data/baseline_spike_train.obj",
                    "rb",
                )
            ).freq
            fr, _ = np.histogram(spikes, bins=bins)
            fr = (fr / bw) / bl_freq if bl_freq > 0 else np.zeros(fr.shape)
            neuron_trials.append(fr)
            if bl_freq > 0:
                neuron_stim.append(stim_freq / bl_freq if bl_freq > 0 else 0)
        all_average.append(np.mean(neuron_trials, axis=0))
        average_stim_norm_fr.append(np.mean(neuron_stim))
    """

    all_average, average_stim_norm_fr = analyze_data.get_norm_frs(
        data, baseline=baseline, stim=stim, post=post, bw=bw
    )

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
    axes.set_xlim([-1, 1.1])
    return average_stim_norm_fr


def psth_norm_compare(
    data1,
    data2,
    axes,
    bw=0.05,
    baseline=2,
    stim=10,
    post=2,
    plot_lims=[-2, 12],
    label1="D1 MSNs",
    label2="GPe-PV",
    color1="orange",
    semcolor1="bisque",
    color2="blue",
    semcolor2="lightblue",
    drawboth=False,
    lw=0.5,
    linestyle="dashed",
):
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

    x1 = np.asarray(norm_fr1)
    x2 = np.asarray(norm_fr2)
    x1 = x1[np.logical_not(np.isnan(x1))]
    x2 = x2[np.logical_not(np.isnan(x2))]
    axes.hlines(np.mean(x1), 0, plot_lims[1], color=color1, ls="dotted", lw=1)
    axes.hlines(np.mean(x2), 0, plot_lims[1], color=color2, ls="dotted", lw=1)

    _, pval = stats.ttest_ind(
        x1, x2, alternative="less" if np.mean(x1) < np.mean(x2) else "greater"
    )
    if pval < 0.001:
        plot_vertical_bracket(
            axes, 1.01, 1.04, np.mean(x1), np.mean(x2), "***", color="k", lw=1
        )
    elif pval < 0.01:
        plot_vertical_bracket(
            axes, 1.01, 1.04, np.mean(x1), np.mean(x2), "**", color="k", lw=1
        )
    elif pval < 0.05:
        plot_vertical_bracket(
            axes, 1.01, 1.04, np.mean(x1), np.mean(x2), "*", color="k", lw=1
        )
    else:
        plot_vertical_bracket(
            axes, 1.01, 1.04, np.mean(x1), np.mean(x2), "n.s.", color="k", lw=1
        )
    axes.set_ylabel("Norm. Firing Rate", fontsize=8)
    axes.set_xlabel("Time (s)", fontsize=8)
    axes.set_xlim([-1, 1.1])


def category_shift_single(
    data1,
    category1,
    category2,
    axes,
    show_shifts=True,
    color1="k",
    tick1="$f_{bl}$",
    tick2="$f_{stim}$",
    title="Comparison",
    ylabel="Firing Rate (Hz)",
    bar=False,
):
    if bar:
        axes.bar(
            [0, 1],
            [np.mean(data1[category1]), np.mean(data1[category2])],
            color=color1,
            zorder=1,
        )
    else:
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

    if not bar:
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

    if show_shifts:
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

    _, pval1 = stats.ttest_rel(data1[category1], data1[category2])
    ylim1 = axes.get_ylim()[1]
    if pval1 < 0.001:
        plot_bracket(axes, 0, 1, ylim1 * 1, ylim1 * 1.05, "+++", lw=1)
    elif pval1 < 0.01:
        plot_bracket(axes, 0, 1, ylim1 * 1, ylim1 * 1.05, "++", lw=1)
    elif pval1 < 0.05:
        plot_bracket(axes, 0, 1, ylim1 * 1, ylim1 * 1.05, "+", lw=1)
    else:
        plot_bracket(axes, 0, 1, ylim1 * 1, ylim1 * 1.05, "n.s.", lw=1)

    axes.set_xticks([0, 1])
    axes.set_xticklabels([tick1, tick2], fontsize=8)
    axes.set_xlim([-0.25, 1.25])
    axes.set_title(title, fontsize=8)
    axes.set_ylabel(ylabel, fontsize=8)
    sns.despine(ax=axes)


def category_shift(
    data1,
    data2,
    category1,
    category2,
    axes,
    show_shifts=True,
    color1="k",
    color2="k",
    tick1="$f_{bl}$",
    tick2="$f_{stim}$",
    title="Comparison",
    ylabel="Firing Rate (Hz)",
    bar=False,
):
    if bar:
        axes.bar(
            [0, 1],
            [np.mean(data1[category1]), np.mean(data1[category2])],
            color=color1,
            zorder=1,
        )
        axes.bar(
            [2, 3],
            [np.mean(data2[category1]), np.mean(data2[category2])],
            color=color2,
            zorder=1,
        )
    else:
        axes.scatter(
            [0, 1],
            [np.mean(data1[category1]), np.mean(data1[category2])],
            color=color1,
            zorder=10,
            s=20,
        )
        axes.scatter(
            [2, 3],
            [np.mean(data2[category1]), np.mean(data2[category2])],
            color=color2,
            zorder=10,
            s=20,
        )
        axes.plot(
            [0, 1],
            [np.mean(data1[category1]), np.mean(data1[category2])],
            color=color1,
            zorder=10,
        )
        axes.plot(
            [2, 3],
            [np.mean(data2[category1]), np.mean(data2[category2])],
            color=color2,
            zorder=10,
        )

    if not bar:
        axes.errorbar(
            0,
            np.mean(data1[category1]),
            yerr=np.std(data1[category1]),
            color=color1,
            zorder=10,
        )
        axes.errorbar(
            1,
            np.mean(data1[category2]),
            yerr=np.std(data1[category2]),
            color=color1,
            zorder=10,
        )
        axes.errorbar(
            2,
            np.mean(data2[category1]),
            yerr=np.std(data2[category1]),
            color=color2,
            zorder=10,
        )
        axes.errorbar(
            3,
            np.mean(data2[category2]),
            yerr=np.std(data2[category2]),
            color=color2,
            zorder=10,
        )

    if show_shifts:
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

        for _, row in data2.iterrows():
            axes.plot(
                [2, 3],
                [row[category1], row[category2]],
                marker="o",
                color="gray",
                markersize=0.5,
                lw=0.25,
                zorder=5,
                alpha=0.25,
            )

    _, pval1 = stats.ttest_rel(data1[category1], data1[category2])
    ylim1 = axes.get_ylim()[1]
    if pval1 < 0.001:
        plot_bracket(axes, 0, 1, ylim1 * 1, ylim1 * 1.05, "+++", lw=1)
    elif pval1 < 0.01:
        plot_bracket(axes, 0, 1, ylim1 * 1, ylim1 * 1.05, "++", lw=1)
    elif pval1 < 0.05:
        plot_bracket(axes, 0, 1, ylim1 * 1, ylim1 * 1.05, "+", lw=1)

    _, pval2 = stats.ttest_rel(data2[category1], data2[category2])
    if pval2 < 0.001:
        plot_bracket(axes, 2, 3, ylim1, ylim1 * 1.05, "+++", lw=1)
    elif pval2 < 0.01:
        plot_bracket(axes, 2, 3, ylim1, ylim1 * 1.05, "++", lw=1)
    elif pval2 < 0.05:
        plot_bracket(axes, 2, 3, ylim1, ylim1 * 1.05, "+", lw=1)

    _, pval3 = stats.ttest_ind(
        data1[category1],
        data2[category1],
        alternative=(
            "less"
            if np.mean(data1[category1]) < np.mean(data2[category1])
            else "greater"
        ),
    )
    if pval3 < 0.001:
        plot_bracket(axes, 0, 2, ylim1 * 1.15, ylim1 * 1.2, "***", lw=1)
    elif pval3 < 0.01:
        plot_bracket(axes, 0, 2, ylim1 * 1.15, ylim1 * 1.2, "**", lw=1)
    elif pval3 < 0.05:
        plot_bracket(axes, 0, 2, ylim1 * 1.15, ylim1 * 1.2, "*", lw=1)

    _, pval4 = stats.ttest_ind(
        data1[category2],
        data2[category2],
        alternative=(
            "less"
            if np.mean(data1[category2]) < np.mean(data2[category2])
            else "greater"
        ),
    )
    if pval4 < 0.001:
        plot_bracket(axes, 1, 3, ylim1 * 1.4, ylim1 * 1.45, "***", lw=1)
    elif pval4 < 0.01:
        plot_bracket(axes, 1, 3, ylim1 * 1.4, ylim1 * 1.45, "**", lw=1)
    elif pval4 < 0.05:
        plot_bracket(axes, 1, 3, ylim1 * 1.4, ylim1 * 1.45, "*", lw=1)

    axes.set_xticks([0, 1, 2, 3])
    axes.set_xticklabels([tick1, tick2, tick1, tick2], fontsize=8)
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
    ylabel="$f_{bl}/f_{stim}$",
    title="All Responses",
    show_scatter=False,
    ylims=[-0.5, 0.5],
):

    y1 = np.zeros(len(data1))
    count = 0
    for _, row in data1.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
        frs = []
        for trial in range(neuron.trials):
            baseline_freq = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/baseline_data/baseline_spike_train.obj",
                    "rb",
                )
            ).freq
            stimulus_freq = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/stimulus_data/stimulus_spike_train.obj",
                    "rb",
                )
            ).freq
            baseline_cv = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/baseline_data/baseline_spike_train.obj",
                    "rb",
                )
            ).cv
            stimulus_cv = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/stimulus_data/stimulus_spike_train.obj",
                    "rb",
                )
            ).cv
            if "freq" in category1 and (stimulus_freq + baseline_freq > 0):
                frs.append(
                    (stimulus_freq - baseline_freq) / (stimulus_freq + baseline_freq)
                )
            elif "cv" in category1 and (stimulus_cv + baseline_cv > 0):
                frs.append((stimulus_cv - baseline_cv) / (stimulus_cv + baseline_cv))
        y1[count] = np.mean(frs)
        count += 1

    y2 = np.zeros(len(data2))
    count = 0
    for index, row in data2.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
        frs = []
        for trial in range(neuron.trials):
            baseline_freq = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/baseline_data/baseline_spike_train.obj",
                    "rb",
                )
            ).freq
            stimulus_freq = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/stimulus_data/stimulus_spike_train.obj",
                    "rb",
                )
            ).freq
            baseline_cv = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/baseline_data/baseline_spike_train.obj",
                    "rb",
                )
            ).cv
            stimulus_cv = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/stimulus_data/stimulus_spike_train.obj",
                    "rb",
                )
            ).cv
            if "freq" in category1 and (stimulus_freq + baseline_freq > 0):
                frs.append(
                    (stimulus_freq - baseline_freq) / (stimulus_freq + baseline_freq)
                )
            elif "cv" in category1 and (stimulus_cv + baseline_cv > 0):
                frs.append((stimulus_cv - baseline_cv) / (stimulus_cv + baseline_cv))
        y2[count] = np.mean(frs)
        count += 1

    y1 = y1[~np.isnan(y1)]
    y2 = y2[~np.isnan(y2)]

    a1 = 0.1
    b1 = 0.15
    a2 = 0.18
    b2 = a2 + (b1 - a1)
    x1 = np.random.rand(len(y1)) * (b1 - a1) + a1
    x2 = np.random.rand(len(y2)) * (b2 - a2) + a2
    if show_scatter:
        axes.scatter(x1, y1, color="gray", s=2, alpha=0.5)
        axes.scatter(x2, y2, color="gray", s=2, alpha=0.5)
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

    _, pval = stats.ttest_ind(
        y1, y2, alternative="less" if np.mean(y1) < np.mean(y2) else "greater"
    )
    ybracket1 = np.abs(ylims[1] - ylims[0]) * 0.05
    ybracket2 = np.abs(ylims[1] - ylims[0]) * 0.1
    if pval < 0.001:
        plot_bracket(
            axes,
            (a1 + b1) / 2,
            (a2 + b2) / 2,
            ybracket1,
            ybracket2,
            "***",
            lw=1,
        )
    elif pval < 0.01:
        plot_bracket(
            axes,
            (a1 + b1) / 2,
            (a2 + b2) / 2,
            ybracket1,
            ybracket2,
            "**",
            lw=1,
        )
    elif pval < 0.05:
        plot_bracket(
            axes,
            (a1 + b1) / 2,
            (a2 + b2) / 2,
            ybracket1,
            ybracket2,
            "*",
            lw=1,
        )
    else:
        plot_bracket(
            axes,
            (a1 + b1) / 2,
            (a2 + b2) / 2,
            ybracket1,
            ybracket2,
            "n.s.",
            lw=1,
        )

    axes.set_ylim(ylims)
    axes.set_xticks([np.mean(x1), np.mean(x2)])
    axes.set_xticklabels([tick1, tick2], fontsize=8)
    axes.set_ylabel(ylabel, fontsize=8)
    xlims = axes.get_xlim()
    axes.hlines(0, xlims[0], xlims[1], color="gray", ls="dashed", lw=0.5)

    axes.set_title(title, fontsize=8)
    sns.despine(ax=axes)


def locations_plot_d1(data, axes):
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
    locs = np.zeros((3, 5))
    for response in range(5):
        locs[0, response] = len(
            data[
                (data["distance"] == 1.1) & (data["neural_response"] == types[response])
            ]
        )
        locs[1, response] = len(
            data[
                (data["distance"] == 1.4) & (data["neural_response"] == types[response])
            ]
        )
        locs[2, response] = len(
            data[
                (data["distance"] == 1.7) & (data["neural_response"] == types[response])
            ]
        )

    counts = [np.sum(locs[i, :], dtype=int) for i in range(3)]
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
    # axes.set_xlim([0,3])
    sns.despine(ax=axes)


def locations_plot_gpe(data, axes):
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
        "biphasic IE",
        "biphasic EI",
    ]
    locs = np.zeros((3, 5))
    for response in range(5):
        locs[0, response] = len(
            data[
                (data["distance"] < 1.4) & (data["neural_response"] == types[response])
            ]
        )
        locs[1, response] = len(
            data[
                (data["distance"] == 1.4) & (data["neural_response"] == types[response])
            ]
        )
        locs[2, response] = len(
            data[
                (data["distance"] > 1.4) & (data["neural_response"] == types[response])
            ]
        )
    counts = [np.sum(locs[i, :], dtype=int) for i in range(3)]
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
    # axes.set_xlim([0,3])
    sns.despine(ax=axes)


def histplot(
    axes,
    x1,
    x2,
    bins,
    xlabel="Baseline FR (Hz)",
    color1="blue",
    color2="gray",
    label1="",
    label2="",
):
    sns.histplot(
        x1,
        bins=bins,
        edgecolor="white",
        lw=0.5,
        stat="probability",
        color=color1,
        ax=axes,
        label=label1,
    )
    sns.histplot(
        x2,
        bins=bins,
        edgecolor="white",
        lw=0.5,
        stat="probability",
        color=color2,
        ax=axes,
        label=label2,
    )

    ylims = axes.get_ylim()
    axes.vlines(np.mean(x1), ylims[0], ylims[1], color=color1, ls="dashed", lw=0.5)
    axes.vlines(np.mean(x2), ylims[0], ylims[1], color=color2, ls="dashed", lw=0.5)

    _, pval3 = stats.ttest_ind(
        x1,
        x2,
        alternative="less" if np.mean(x1) < np.mean(x2) else "greater",
    )
    if pval3 < 0.001:
        plot_bracket(
            axes, np.mean(x1), np.mean(x2), ylims[1] * 1.15, ylims[1] * 1.2, "***", lw=1
        )
    elif pval3 < 0.01:
        plot_bracket(
            axes, np.mean(x1), np.mean(x2), ylims[1] * 1.15, ylims[1] * 1.2, "**", lw=1
        )
    elif pval3 < 0.05:
        plot_bracket(
            axes, np.mean(x1), np.mean(x2), ylims[1] * 1.15, ylims[1] * 1.2, "*", lw=1
        )

    axes.set_xlabel(xlabel)
    if label1 != "" or label2 != "":
        axes.legend(frameon=False, fancybox=False)
    sns.despine(ax=axes)


def plot_bracket(ax, x1, x2, y1, y2, text, color="k", lw=1):
    ax.hlines(y2, x1, x2, color=color, lw=lw, zorder=15)
    ax.vlines([x1, x2], y1, y2, color=color, lw=lw, zorder=15)
    ax.annotate(
        text,
        xycoords="data",
        xy=((x1 + x2) / 2, y2 + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01),
        fontsize=12 if text != "n.s." else 8,
        zorder=15,
        ha="center",
    )


def plot_vertical_bracket(ax, x1, x2, y1, y2, text, color="k", lw=1):
    ax.hlines([y1, y2], x1, x2, color=color, lw=lw, zorder=15)
    ax.vlines(x2, y1, y2, color=color, lw=lw, zorder=15)
    ax.annotate(
        text,
        xycoords="data",
        xy=(x2 + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01, (y1 + y2) / 2),
        fontsize=12 if text != "n.s." else 8,
        zorder=15,
        va="center",
    )
