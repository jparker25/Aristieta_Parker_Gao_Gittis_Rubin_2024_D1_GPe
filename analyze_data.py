import numpy as np
import scipy.stats as stats
import pickle
import pandas as pd
from matplotlib import pyplot as plt


def get_delta_category(data, category):
    deltas = np.zeros(len(data))
    count = 0
    for _, row in data.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
        frs = []
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

            stim_cv = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/stimulus_data/stimulus_spike_train.obj",
                    "rb",
                )
            ).cv

            base_cv = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/baseline_data/baseline_spike_train.obj",
                    "rb",
                )
            ).cv

            if "cv" in category:
                frs.append(stim_cv - base_cv)
            if "freq" in category:
                frs.append(stim_rate - base_rate)
        deltas[count] = np.mean(frs)
        count += 1
    return deltas[~np.isnan(deltas)]


def get_all_deltas(data, category):
    deltas = []
    for _, row in data.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
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

            stim_cv = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/stimulus_data/stimulus_spike_train.obj",
                    "rb",
                )
            ).cv

            base_cv = pickle.load(
                open(
                    f"{src}/trial_{trial+1:02d}/baseline_data/baseline_spike_train.obj",
                    "rb",
                )
            ).cv

            if "cv" in category:
                deltas.append(stim_cv - base_cv)
            if "freq" in category:
                deltas.append(stim_rate - base_rate)
    return deltas


def get_all_modulation_factors(data):
    types = [
        "no effect",
        "complete inhibition",
        "partial inhibition",
        "adapting inhibition",
        "excitation",
    ]
    modulation_factors = []
    for response in types:
        subset = data[data["neural_response"] == response]
        for _, row in subset.iterrows():
            src = row["cell_dir"]
            neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
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

    return modulation_factors


def get_modulation_factors(data, category):
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
                if "freq" in category and (stim_rate + base_rate > 0):
                    modulation_factors.append(
                        (stim_rate - base_rate) / (stim_rate + base_rate)
                    )
                elif "cv" in category and (stimulus_cv + baseline_cv > 0):
                    modulation_factors.append(
                        (stimulus_cv - baseline_cv) / (stimulus_cv + baseline_cv)
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

    inh_mfs = np.asarray(inh_mfs)
    ex_mfs = np.asarray(ex_mfs)
    ne_mfs = np.asarray(ne_mfs)
    all_rates = all_rates[~np.isnan(all_rates).any(axis=1)]
    return all_rates, inh_mfs, ex_mfs, ne_mfs


def get_norm_frs(data, baseline=2, stim=1, post=0, bw=0.05):
    bins = np.arange(-baseline, stim + post + bw, bw)
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
            if stim_freq == 0:
                neuron_stim.append(0)
            elif bl_freq > 0:
                neuron_stim.append(stim_freq / bl_freq)
        all_average.append(np.mean(neuron_trials, axis=0))
        average_stim_norm_fr.append(np.mean(neuron_stim))
    """
    average_stim_norm_fr = []
    for _, row in data.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
        neuron_stim = []
        for trial in range(neuron.trials):
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
            neuron_stim.append(stim_freq / bl_freq if bl_freq > 0 else 0)
        average_stim_norm_fr.append(np.mean(neuron_stim))
    return average_stim_norm_fr
    """
    return all_average, average_stim_norm_fr


def get_gpe_locations(data):
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
    return locs


def get_d1_locations(data):
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
    return locs


def get_response_distribution(data):
    types = [
        "no effect",
        "complete inhibition",
        "partial inhibition",
        "adapting inhibition",
        "excitation",
    ]
    responses = np.zeros(len(types))
    for i in range(len(types)):
        responses[i] += len(data[data["neural_response"] == types[i]])
    return responses
