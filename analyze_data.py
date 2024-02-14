"""
File: analyze_data.py
Author: John E. Parker
Date: 14 February 2024

Description: Gathers data values for plotting and statistical results.
"""

import numpy as np
import pickle


# Global variable, default types to iterate through
types = [
    "no effect",
    "complete inhibition",
    "partial inhibition",
    "adapting inhibition",
    "excitation",
]


def get_delta_category(data, category):
    """
    Calculates all delta values (stim - pre-stim) for given category in cells in data.

    data\t:\tdata set to compute deltas
    category\t:\tcategory to compute deltas (freq of cv)
    """

    # Iterate through all cells in data and calculate delta
    deltas = np.zeros(len(data))
    count = 0
    for _, row in data.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
        frs = []
        for trial in range(neuron.trials):

            # Grab approrpriate data for stim and baseline categories
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

            # Compute and store delta
            if "cv" in category:
                frs.append(stim_cv - base_cv)
            if "freq" in category:
                frs.append(stim_rate - base_rate)
        deltas[count] = np.mean(frs)
        count += 1

    # Remove NANs and return result
    return deltas[~np.isnan(deltas)]


def get_all_deltas(data, category):
    """
    Calculates all delta values (stim - pre-stim) for given category in all trials in data.

    data\t:\tdata set to compute deltas
    category\t:\tcategory to compute deltas (freq of cv)
    """
    # Iterate through all trials in data and calculate delta
    deltas = []
    for _, row in data.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
        for trial in range(neuron.trials):

            # Grab approrpiate categorical data
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

            # compute delta
            if "cv" in category:
                deltas.append(stim_cv - base_cv)
            if "freq" in category:
                deltas.append(stim_rate - base_rate)
    return deltas


def get_all_modulation_factors(data):
    """
    Calculates all modulation factor (stim - baseline / stim + baseline) for FR in all trials in data.

    data\t:\tdata set to compute MFs
    """
    # Iterate through all cells and gather MFs
    modulation_factors = []
    for response in types:
        subset = data[data["neural_response"] == response]
        for _, row in subset.iterrows():
            src = row["cell_dir"]
            neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
            # Iterate through trials and append all MFs
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


def calc_mf(neuron, category):
    """
    Return average categorical MF for a given neuron.

    neuron\t:\tneuron to compute MF for
    category\t:\tcategory to compute MF (cv or freq)
    """

    # Iterate through all trials and collect MFs to average
    modulation_factors = []
    for trial in range(neuron.trials):
        # Grab rates and CVs for stim and baseline
        stim_rate = pickle.load(
            open(
                f"{neuron.cell_dir}/trial_{trial+1:02d}/stimulus_data/stimulus_spike_train.obj",
                "rb",
            )
        ).freq

        base_rate = pickle.load(
            open(
                f"{neuron.cell_dir}/trial_{trial+1:02d}/baseline_data/baseline_spike_train.obj",
                "rb",
            )
        ).freq
        baseline_cv = pickle.load(
            open(
                f"{neuron.cell_dir}/trial_{trial+1:02d}/baseline_data/baseline_spike_train.obj",
                "rb",
            )
        ).cv
        stimulus_cv = pickle.load(
            open(
                f"{neuron.cell_dir}/trial_{trial+1:02d}/stimulus_data/stimulus_spike_train.obj",
                "rb",
            )
        ).cv
        # Append to modulation factors array if not NAN
        if "freq" in category and (stim_rate + base_rate > 0):
            modulation_factors.append((stim_rate - base_rate) / (stim_rate + base_rate))
        elif "cv" in category and (stimulus_cv + baseline_cv > 0):
            modulation_factors.append(
                (stimulus_cv - baseline_cv) / (stimulus_cv + baseline_cv)
            )
    return modulation_factors


def get_modulation_factors(data, category):
    """
    Returns array with pre-exp firing rate and MF FR, and MFs split by inhibition, excitaiton and no effect.

    data\t:\tdata to collect MFs for
    category\t:\tcategory to compute MF (cv or freq)
    """
    # Create arrays to populate
    all_rates = np.zeros((len(data), 3))
    all_count = 0
    ne_mfs = []
    inh_mfs = []
    ex_mfs = []
    type_count = 0
    # Iterate through cells within each type
    for response in types:
        subset = data[data["neural_response"] == response]
        rates = np.zeros((len(subset), 3))
        count = 0
        for _, row in subset.iterrows():
            src = row["cell_dir"]
            neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
            # Calculate moduatoin factors for each cell and sort accordingly into correct arrays
            modulation_factors = calc_mf(neuron, category)
            if len(modulation_factors) > 0:
                rates[count, 0] = row["pre_exp_freq"]
                rates[count, 1] = np.mean(modulation_factors)
                rates[count, 2] = type_count

            if "inhibition" in response:
                inh_mfs.append(rates[count, 1])
            elif "excitation" in response:
                ex_mfs.append(rates[count, 1])
            else:
                ne_mfs.append(rates[count, 1])
            all_rates[all_count, :] = [
                rates[count, 0],
                rates[count, 1],
                rates[count, 2],
            ]
            count += 1
            all_count += 1
        type_count += 1

    # Clean up arrays
    inh_mfs = np.asarray(inh_mfs)
    ex_mfs = np.asarray(ex_mfs)
    ne_mfs = np.asarray(ne_mfs)
    all_rates = all_rates[~np.isnan(all_rates).any(axis=1)]
    return all_rates, inh_mfs, ex_mfs, ne_mfs


def get_norm_frs(data, baseline=2, stim=1, post=0, bw=0.05):
    """
    Returns PSTH with normalized FR over a given bw and the average normalized stim FR for a data set.

    data\t:\tdata to collect norm FRs
    baseline\t:\tlength of baseline to compute norm FR PSTH (default = 2)
    stim\t:\tlength of stim to compute norm FR PSTH (default = 1)
    post\t:\tlength of post to compute norm FR PSTH (default = 0)
    bw\t:\tbinwidth to calculate PSTH (default = 0.05)
    """
    # Create bins to find PSTH
    bins = np.arange(-baseline, stim + post + bw, bw)
    all_average = []
    average_stim_norm_fr = []
    # Iterate through cells in dataset
    for _, row in data.iterrows():
        src = row["cell_dir"]
        neuron = pickle.load(open(f"{src}/neuron.obj", "rb"))
        light_on = np.loadtxt(f"{src}/light_on.txt")
        neuron_trials = []
        neuron_stim = []
        # Iterate through trials and find cell average PSTH and norm FR
        for trial in range(neuron.trials):

            # Grab trial data
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

            # Calculate FR psth and normalize to baseline firing rate
            fr, _ = np.histogram(spikes, bins=bins)
            fr = (fr / bw) / bl_freq if bl_freq > 0 else np.zeros(fr.shape)
            neuron_trials.append(fr)
            if stim_freq == 0:
                neuron_stim.append(0)
            elif bl_freq > 0:
                neuron_stim.append(stim_freq / bl_freq)
        # Append trial average
        all_average.append(np.mean(neuron_trials, axis=0))
        average_stim_norm_fr.append(np.mean(neuron_stim))
    return all_average, average_stim_norm_fr


def get_gpe_locations(data):
    """
    Return array of cells in each location for gpe data (medial, central, lateral).

    data\t:\tdata to find distribution of responses
    """
    locs = np.zeros((3, 5))
    for response in range(5):
        # Place appropriately for position
        # Medial
        locs[0, response] = len(
            data[
                (data["distance"] < 1.4) & (data["neural_response"] == types[response])
            ]
        )
        # Central
        locs[1, response] = len(
            data[
                (data["distance"] == 1.4) & (data["neural_response"] == types[response])
            ]
        )
        # Lateral
        locs[2, response] = len(
            data[
                (data["distance"] > 1.4) & (data["neural_response"] == types[response])
            ]
        )
    return locs


def get_d1_locations(data):
    """
    Return array of cells in each location for d1 data (medial, central, lateral).

    data\t:\tdata to find distribution of responses
    """
    locs = np.zeros((3, 5))
    for response in range(5):
        # Place appropriately for position
        # Medial
        locs[0, response] = len(
            data[
                (data["distance"] == 1.1) & (data["neural_response"] == types[response])
            ]
        )
        # Central
        locs[1, response] = len(
            data[
                (data["distance"] == 1.4) & (data["neural_response"] == types[response])
            ]
        )  # Lateral
        locs[2, response] = len(
            data[
                (data["distance"] == 1.7) & (data["neural_response"] == types[response])
            ]
        )
    return locs


def get_response_distribution(data):
    """
    Return array of cells in each response for data set.

    data\t:\tdata to find distribution of responses
    """

    # Iterate through types in type and count cells in each type
    responses = np.zeros(len(types))
    for i in range(len(types)):
        responses[i] += len(data[data["neural_response"] == types[i]])
    return responses
