################################################################################
# helpers.py
# Commonly used functions in program that do not belong to a single script.
# Author: John E. Parker
################################################################################

# import python modules
import os, sys, string
import numpy as np

"""
def plot_bracket(ax, x1, x2, y1, y2, text, color="k", lw=0.5):
    ax.hlines(y2, x1, x2, color=color, lw=lw)
    ax.vlines([x1, x2], y1, y2, color=color, lw=lw)
    ax.annotate(
        text,
        xycoords="data",
        xy=((x1 + x2) / 2, y2 + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01),
    )
"""


def run_cmd(str):
    print(str)
    os.system(str)
    print()


def makeNice(axes, labelsize=8):
    if type(axes) == list:
        for ax in axes:
            for i in ["left", "right", "top", "bottom"]:
                if i != "left" and i != "bottom":
                    ax.spines[i].set_visible(False)
                    ax.tick_params("both", width=0, labelsize=labelsize)
                else:
                    ax.spines[i].set_linewidth(3)
                    ax.tick_params("both", width=0, labelsize=labelsize)
    else:
        for i in ["left", "right", "top", "bottom"]:
            if i != "left" and i != "bottom":
                axes.spines[i].set_visible(False)
                axes.tick_params("both", width=0, labelsize=labelsize)
            else:
                axes.spines[i].set_linewidth(3)
                axes.tick_params("both", width=0, labelsize=labelsize)


def add_fig_labels(axes, start_at=0, fontsize=12):
    labels = string.ascii_uppercase
    for i in range(len(axes)):
        axes[i].text(
            -0.15,
            1.05,
            labels[start_at + i],
            fontsize=fontsize,
            transform=axes[i].transAxes,
            fontweight="bold",
            color="k",
        )


def match_axis(axes, type="both"):
    if type == "x":
        min = np.min([ax.get_xlim()[0] for ax in axes])
        max = np.max([ax.get_xlim()[1] for ax in axes])
        for ax in axes:
            ax.set_xlim([min, max])
    elif type == "y":
        min = np.min([ax.get_ylim()[0] for ax in axes])
        max = np.max([ax.get_ylim()[1] for ax in axes])
        for ax in axes:
            ax.set_ylim([min, max])
    else:
        match_axis(axes, type="x")
        match_axis(axes, type="y")
