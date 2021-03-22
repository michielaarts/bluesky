import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scn_reader import create_routing_df
from pathlib import Path
import warnings
import re
from typing import List, Tuple
import pickle as pkl
from tkinter import Tk, filedialog

# Standard inputs.
OUTPUT_FOLDER = Path('../output/')
SCN_FOLDER = Path('../scenario/URBAN/')
PROCESS_WR = True


def process_batch_file(filename: str) -> List[str]:
    """
    Obtains the scenario files called by a batch file.

    :param filename: a batch *.scn file
    :return: a list of *.scn files
    """
    filepath = Path(filename)
    if not filepath.is_file():
        warnings.warn(f'File {filename} does not exist!')
        return []

    with open(filepath, 'r') as f:
        pcall_files = []
        for line in f.readlines():
            if line.startswith('# File:'):
                pcall_file = re.findall('# File: (.*)\n', line)[0]
                pcall_files.append(pcall_file)

    return pcall_files


def create_result_dict(scn_folder: Path = SCN_FOLDER, output_folder: Path = OUTPUT_FOLDER,
                       process_wr: bool = PROCESS_WR) -> dict:
    """
    Creates a result dict of a batch file.
    Also saves that to a pickle

    :param scn_folder: Scenario folder path
    :param output_folder: Output folder path
    :param process_wr: Also process with resolution case
    :return: result dict
    """
    # Load SCN file.
    tk_root = Tk()
    scn_file = filedialog.askopenfilename(initialdir=scn_folder, title='Select a batch or scn file',
                                          filetypes=[('scenario', '*.scn')])
    scenario_name = re.findall('.*(batch_.*).scn', scn_file)[0]

    # Check if batch file.
    result = {'name': scenario_name}
    all_scn_names = []
    if 'batch' in scn_file:
        pcall_files = process_batch_file(scn_file)
        reso_cases = ['NR']
        if process_wr:
            reso_cases.append('WR')
        logs = filedialog.askopenfilenames(initialdir=output_folder, title=f'Select all logs for {scenario_name}',
                                           filetypes=[('log', '*.log')])
        for reso in reso_cases:
            for pcall in pcall_files:
                scn_name = f'{reso}_{pcall[:-4]}'
                all_scn_names.append(scn_name)
                result[scn_name] = {'scn_file': pcall}
                for log in logs:
                    if scn_name.lower() in log.lower():
                        if 'CONFLOG' in log:
                            result[scn_name]['conflogfile'] = Path(log)
                        elif 'FLSTLOG' in log:
                            result[scn_name]['flstlogfile'] = Path(log)
                        else:
                            raise ValueError('Select flstlog or conflogs')
    else:
        raise NotImplementedError('Non-batch files not yet implemented')
    tk_root.destroy()

    pop_runs = []
    for run in result.keys():
        if run == 'name':
            continue
        # Read scenario file.
        with open(scn_folder / 'Data' / (run[3:] + '.pkl'), 'rb') as scn_data_file:
            result[run]['scn'] = pkl.load(scn_data_file)

        # Read log files.
        if 'conflogfile' in result[run].keys():
            result[run]['conflog'] = pd.read_csv(result[run]['conflogfile'],
                                                 comment='#', skipinitialspace=True)
        else:
            print(f'Could not find conflogfile for scn {run}')
            pop_runs.append(run)

        if 'flstlogfile' in result[run].keys():
            result[run]['flstlog'] = pd.read_csv(result[run]['flstlogfile'],
                                                 comment='#', skipinitialspace=True)
        else:
            print(f'Could not find flstlogfile for scn {run}')
            pop_runs.append(run)

    for run in np.unique(pop_runs):
        # Pop run from result dict.
        result.pop(run)
    return result


def save_result(result: dict, output_folder: Path = OUTPUT_FOLDER) -> None:
    """
    Saves the result to a pickle file.

    :param result: result dict
    :param output_folder: Path to output folder
    :return: None
    """
    # Backup result to pickle.
    result_folder = output_folder / 'RESULT' / ''
    if not result_folder.is_dir():
        result_folder.mkdir(parents=True, exist_ok=True)
    result_save_name = result_folder / (result['name'] + '.pkl')
    if result_save_name.is_file():
        while True:
            print(f'File {result_save_name} already exists! Overwrite? [y/n]')
            choice = input().lower()
            if choice == 'y':
                save = True
                break
            elif choice == 'n':
                save = False
                break
            else:
                print('Invalid choice, please type y or n')
    else:
        save = True
    if save:
        with open(result_save_name, 'wb') as f:
            pkl.dump(result, f)


def process_result(result: dict) -> dict:
    """
    Processes the conflog and the flstlog and adds to the result dict.

    :param result: result dict from create_result_dict()
    :return: extended result dict
    """

    for run in sorted(result.keys()):
        # First process the NR cases, then the WR cases.
        if run == 'name':
            continue
        # Extract logging times.
        logging_start = result[run]['conflog']['t'][0] + result[run]['scn']['duration'][0]
        logging_end = logging_start + result[run]['scn']['duration'][1]
        result[run]['logging_time'] = (logging_start, logging_end)

        # Process logs.
        result[run]['conf'] = process_conflog(result[run]['conflog'], logging_start, logging_end)
        if run.startswith('NR'):
            # NR case.
            result[run]['flst'], result[run]['ac'] = process_flstlog(result[run]['flstlog'],
                                                                     logging_start, logging_end)
        else:
            # WR case.
            nr_run = 'NR_' + run[3:]
            result[run]['ac'] = result[nr_run]['ac']
            result[run]['flst'], _ = process_flstlog(result[run]['flstlog'],
                                                     logging_start, logging_end,
                                                     result[run]['ac'])
    return result


def process_conflog(conf_df: pd.DataFrame, start_time: float, end_time: float) -> dict:
    """
    Extracts the mean and total variables during the logging period from the CONFLOG.

    :param conf_df: CONFLOG dataframe
    :param start_time: Start time of logging period
    :param end_time: End time of logging period
    :return: Dict with conflict values
    """
    logging_df = conf_df[(conf_df['t'] >= start_time) & (conf_df['t'] < end_time)]
    ni = logging_df[['ni_ac', 'ni_conf', 'ni_los']].mean()
    ntotal = (logging_df[['ntotal_ac', 'ntotal_conf', 'ntotal_los']].iloc[-1] -
              logging_df[['ntotal_ac', 'ntotal_conf', 'ntotal_los']].iloc[0])
    cr = bool(logging_df['cr'].iloc[0])
    conf = ni.to_dict()
    conf.update(ntotal.to_dict())
    conf['cr'] = cr

    # Sanity check if all scenarios are completely cooled down.
    if conf_df['ni_ac'].iloc[-1] != 0:
        print(f"WARNING: Scenario for N_inst={ni}, with CR O{'N' if cr else 'FF'} did not completely cool down!")

    return conf


def process_flstlog(flst_df: pd.DataFrame, start_time: float, end_time: float, ac=None) -> Tuple[dict, np.ndarray]:
    """
    Extracts the mean and total variables during the logging period of the FLSTLOG.

    :param flst_df: FLSTLOG dataframe
    :param start_time: Start time of logging period
    :param end_time: End time of logging period
    :param ac: List of aircraft extracted in the NR FLSTLOG logging period
    :return: Tuple with [flst dict, list of ac extracted]
    """
    if ac is not None:
        # WR case.
        logging_df = flst_df[flst_df['callsign'].isin(ac)]
        flst = logging_df[['flight_time', 'dist2D', 'dist3D', 'work_done', 'tas']].mean().to_dict()
        flst['num_ac'] = len(ac)
        return flst, ac
    else:
        # NR case.
        logging_df = flst_df[(flst_df['departure_time'] >= start_time) & (flst_df['departure_time'] < end_time)]
        all_ac = logging_df['callsign'].to_numpy()
        flst = logging_df[['flight_time', 'dist2D', 'dist3D', 'work_done', 'tas']].mean().to_dict()
        flst['num_ac'] = len(all_ac)
        return flst, all_ac


def plot_result(result: dict) -> List[plt.Figure]:
    """
    Plots the results.

    :param result: dict from process_result
    :return: List with conf_fig and flst_fig handles
    """
    # Initialize plots.
    conf_fig, conf_axs = plt.subplots(2, 3, num=1)
    plt.get_current_fig_manager().window.showMaximized()
    flst_fig, flst_axs = plt.subplots(1, 3, num=2)
    plt.get_current_fig_manager().window.showMaximized()
    conf_axs = conf_axs.flatten()
    flst_axs = flst_axs.flatten()

    # Extract data.
    data = {}
    for run in result.keys():
        if run == 'name':
            continue
        if run.startswith('NR'):
            reso = 'NR'
        else:
            reso = 'WR'
        if reso not in data.keys():
            data[reso] = {}
            # First run.
            data[reso] = result[run]['conf']
            data[reso].update(result[run]['flst'])
            for key in data[reso].keys():
                data[reso][key] = [data[reso][key]]
        else:
            # Append other runs to lists.
            for conf_key in result[run]['conf'].keys():
                data[reso][conf_key].append(result[run]['conf'][conf_key])
            for flst_key in result[run]['flst'].keys():
                data[reso][flst_key].append(result[run]['flst'][flst_key])

    # Plot values.
    for reso in data.keys():
        if reso == 'NR':
            color = 'blue'
        else:
            color = 'red'

        x = data[reso]['ni_ac']
        conf_axs[0].scatter(x, data[reso]['ni_conf'], color=color, label=reso)
        conf_axs[0].set_ylabel('Inst. no. of conflicts [-]')
        conf_axs[1].scatter(x, data[reso]['ni_los'], color=color, label=reso)
        conf_axs[1].set_ylabel('Inst. no. of los [-]')
        conf_axs[2].scatter(x, data[reso]['ntotal_ac'], color=color, label=reso)
        conf_axs[2].set_ylabel('Total no. of A/C [-]')
        conf_axs[3].scatter(x, data[reso]['ntotal_conf'], color=color, label=reso)
        conf_axs[3].set_ylabel('Total no. of conflicts [-]')
        conf_axs[4].scatter(x, data[reso]['ntotal_los'], color=color, label=reso)
        conf_axs[4].set_ylabel('Total no. of los [-]')
        for ax in conf_axs:
            ax.set_xlabel('Inst. no. of aircraft [-]')
            ax.legend()

        flst_axs[0].scatter(x, data[reso]['flight_time'], color=color, label=reso)
        flst_axs[0].set_ylabel('Mean flight time [s]')
        flst_axs[1].scatter(x, data[reso]['dist3D'], color=color, label=reso)
        flst_axs[1].set_ylabel('Mean 3D distance [m]')
        flst_axs[2].scatter(x, np.array(data[reso]['dist3D']) / np.array(data[reso]['flight_time']),
                            color=color, label=reso)
        flst_axs[2].set_ylabel('Mean velocity [m/s]')
        for ax in flst_axs:
            ax.set_xlabel('Inst. no. of aircraft [-]')
            ax.legend()

    return [conf_fig, flst_fig]


def save_plots(fig_list: List[plt.figure], name: str, output_dir: Path = OUTPUT_FOLDER) -> None:
    """
    Saves the figures

    :param fig_list: List with conf_fig and flst_fig
    :param name: save name
    :param output_dir:
    :return: None
    """
    for fig in fig_list:
        fig.savefig(output_dir / 'RESULT' / f'{name}_{fig.number}.svg', bbox_inches='tight')


if __name__ == '__main__':
    res = create_result_dict()
    res = process_result(res)
    save_result(res)

    # res_pkl = Path(r'C:\Users\michi\OneDrive\Documenten\GitHub\bluesky\output\RESULT\batch_urban_grid_NR.pkl')
    # with open(res_pkl, 'rb') as f:
    #     res = pkl.load(f)

    figs = plot_result(res)
    save_plots(figs, res['name'])
