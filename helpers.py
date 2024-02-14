"""
File: helpers.py
Author: John E. Parker
Date: 14 February 2024

Description: Miscellaneous helper scripts.
"""

# import python modules
import os, string
import numpy as np


def run_cmd(str):
    """
    Prints then runs command on terminal.

    str\t:\tcommand to be run on terminal
    """
    print(str)
    os.system(str)
    print()


def makeNice(axes, labelsize=8):
    """
    Removes left and bottom spines, adjusts labels.

    axes\t:\tfigure axes to be adjusted
    labelsize\t:\tfontsize for tick labels (default = 8)
    """

    # check is list of axes or not, then remove spines and adjust labels
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
    """
    Adds alphabetical figure labels to axes.

    axes\t:\tfigure axes to be adjusted
    start_at\t:\tindex to start alpabetical count (default = 0)
    fontsize\t:\tfontsize for figure labels (default = 12)
    """

    # Iterate through axes and add alphabetical label
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
    """
    Matches horizontal and/or vertical axis limits list of axes.

    axes\t:\tlist of axes to match axis
    type\t:\tlabel to specify which axis to match (defaul = "both")
    """

    # Match only x-axis
    if type == "x":
        min = np.min([ax.get_xlim()[0] for ax in axes])
        max = np.max([ax.get_xlim()[1] for ax in axes])
        for ax in axes:
            ax.set_xlim([min, max])

    # Match only y-axis
    elif type == "y":
        min = np.min([ax.get_ylim()[0] for ax in axes])
        max = np.max([ax.get_ylim()[1] for ax in axes])
        for ax in axes:
            ax.set_ylim([min, max])
    # Match x and y axis
    else:
        match_axis(axes, type="x")
        match_axis(axes, type="y")
