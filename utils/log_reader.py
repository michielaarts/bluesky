import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import re
from typing import List, Tuple
import pickle as pkl
from tkinter import Tk, filedialog
from analytical import AnalyticalModel
from plugins.urban import UrbanGrid

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
    conf['stable'] = all(logging_df['stable'])

    # Sanity check if scenario is completely cooled down.
    if conf_df['ni_ac'].iloc[-1] != 0:
        print(f"WARNING: Scenario for N_inst={ni[0]:.1f}, with CR O{'N' if cr else 'FF'} did not completely cool down!")
        conf['cooled_down'] = False
    else:
        conf['cooled_down'] = True

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


def load_analytical_model(result: dict, scn_folder: Path = SCN_FOLDER) -> Tuple[UrbanGrid, AnalyticalModel]:
    prefix = re.findall('batch_(.*)_NR', result['name'])[0]
    grid_pkl = scn_folder / 'Data' / f'{prefix}_urban_grid.pkl'
    with open(grid_pkl, 'rb') as f:
        urban_grid = pkl.load(f)

    # Extract parameters for analytical model.
    all_runs = [run for run in result.keys() if run != 'name']
    all_speeds = [result[run]['scn']['speed'] for run in all_runs]
    all_s_h = [result[run]['scn']['s_h'] for run in all_runs]
    all_s_v = [result[run]['scn']['s_v'] for run in all_runs]
    all_t_l = [result[run]['scn']['t_l'] for run in all_runs]
    max_val = max([result[run]['scn']['n_inst'] for run in all_runs])
    duration = result[all_runs[0]]['scn']['duration']

    if np.any([len(np.unique(var)) > 1 for var in (all_speeds, all_s_h, all_s_v, all_t_l)]):
        raise NotImplementedError('Implement multiple analytical models in log_reader')
    else:
        speed = all_speeds[0]
        s_h = all_s_h[0]
        s_v = all_s_v[0]
        t_l = all_t_l[0]
    ana_model = AnalyticalModel(urban_grid, max_value=max_val * 1.1, accuracy=25,
                                duration=duration, speed=speed, s_h=s_h, s_v=s_v, t_l=t_l)
    return urban_grid, ana_model


def plot_result(result: dict, ana_model: AnalyticalModel) -> Tuple[List[plt.Figure], dict]:
    """
    Plots the results.

    :param result: dict from process_result
    :param ana_model: Analytical model
    :return: (List with conf_fig and flst_fig handles, data dict)
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

    # Transform to arrays.
    for reso in data.keys():
        for key in data[reso].keys():
            data[reso][key] = np.array(data[reso][key])
        data[reso]['mean_v'] = data[reso]['dist3D'] / data[reso]['flight_time']
        data[reso]['stable_filter'] = data[reso]['stable'] & data[reso]['cooled_down']
        # Set all unstable data to zero.
        for key in data[reso].keys():
            if key != 'stable_filter':
                data[reso][key] = np.where(data[reso]['stable_filter'], data[reso][key], 0)

    # Plot stable values.
    x = data['NR']['ni_ac']
    for reso in data.keys():
        if reso == 'NR':
            color = 'blue'
        else:
            color = 'red'

        stable_filter = data[reso]['stable_filter']

        conf_axs[0].scatter(x[stable_filter], data[reso]['ni_conf'][stable_filter], color=color, label=reso)
        conf_axs[1].scatter(x[stable_filter], data[reso]['ni_los'][stable_filter], color=color, label=reso)
        conf_axs[2].scatter(x[stable_filter], data[reso]['ni_ac'][stable_filter], color=color, label=reso)
        conf_axs[3].scatter(x[stable_filter], data[reso]['ntotal_conf'][stable_filter], color=color, label=reso)
        conf_axs[4].scatter(x[stable_filter], data[reso]['ntotal_los'][stable_filter], color=color, label=reso)
        conf_axs[5].scatter(x[stable_filter], data[reso]['ntotal_ac'][stable_filter], color=color, label=reso)

        flst_axs[0].scatter(x[stable_filter], data[reso]['flight_time'][stable_filter], color=color, label=reso)
        flst_axs[1].scatter(x[stable_filter], data[reso]['dist2D'][stable_filter], color=color, label=reso)
        flst_axs[2].scatter(x[stable_filter], data[reso]['mean_v'][stable_filter], color=color, label=reso)

    # Plot unstable values.
    stable_filter = data['WR']['stable_filter']
    color = 'red'
    conf_axs[0].plot(x[~stable_filter], data[reso]['ni_conf'][~stable_filter],
                     '*', color=color, label=f'{reso}, unstable')
    conf_axs[1].plot(x[~stable_filter], data[reso]['ni_los'][~stable_filter],
                     '*', color=color, label=f'{reso}, unstable')
    conf_axs[2].plot(x[~stable_filter], data[reso]['ni_ac'][~stable_filter],
                     '*', color=color, label=f'{reso}, unstable')
    conf_axs[3].plot(x[~stable_filter], data[reso]['ntotal_conf'][~stable_filter],
                     '*', color=color, label=f'{reso}, unstable')
    conf_axs[4].plot(x[~stable_filter], data[reso]['ntotal_los'][~stable_filter],
                     '*', color=color, label=f'{reso}, unstable')
    conf_axs[5].plot(x[~stable_filter], data[reso]['ntotal_ac'][~stable_filter],
                     '*', color=color, label=f'{reso}, unstable')

    flst_axs[0].plot(x[~stable_filter], data[reso]['flight_time'][~stable_filter],
                     '*', color=color, label=f'{reso}, unstable')
    flst_axs[1].plot(x[~stable_filter], data[reso]['dist2D'][~stable_filter],
                     '*', color=color, label=f'{reso}, unstable')
    flst_axs[2].plot(x[~stable_filter], data[reso]['mean_v'][~stable_filter],
                     '*', color=color, label=f'{reso}, unstable')

    # Fit and plot analytical model derivatives.
    ana_model.fit_derivatives(data)

    # conf_ylim = {ax: ax.get_ylim() for ax in conf_axs}
    conf_axs[0].set_ylabel('Inst. no. of conflicts [-]')
    conf_axs[0].plot(ana_model.n_inst, ana_model.c_inst_nr, color='blue', label='NR Model')
    conf_axs[0].plot(ana_model.n_inst, ana_model.c_inst_wr_fitted, color='coral', linestyle='--', label='WR Fitted')
    conf_axs[1].set_ylabel('Inst. no. of los [-]')
    conf_axs[1].plot(ana_model.n_inst, ana_model.los_inst_nr, color='blue', label='NR Model')
    conf_axs[1].plot(ana_model.n_inst, ana_model.los_inst_nr_fitted, color='lightblue', linestyle='--',
                     label=rf'NR Fitted, $\bar{{t_{{los,NR}}}}={ana_model.avg_los_duration_nr:.1f}$s')
    conf_axs[1].plot(ana_model.n_inst, ana_model.los_inst_wr, color='coral', linestyle='--',
                     label=rf'WR Fitted, $\bar{{t_{{los,WR}}}}={ana_model.avg_los_duration_wr:.1f}$s')
    conf_axs[2].set_ylabel('WR Inst. no. of aircraft')
    conf_axs[2].plot(ana_model.n_inst, ana_model.n_inst, color='blue', label='NR Model')
    conf_axs[2].plot(ana_model.n_inst, ana_model.n_inst_wr, color='red', label='WR Model')
    conf_axs[3].set_ylabel('Total no. of conflicts [-]')
    conf_axs[3].plot(ana_model.n_inst, ana_model.c_total_nr, color='lightblue', linestyle='--',
                     label=rf'NR Fitted, $\bar{{t_{{c,NR}}}}={ana_model.avg_conflict_duration_nr:.1f}$s')
    conf_axs[3].plot(ana_model.n_inst, ana_model.c_total_wr, color='coral', linestyle='--',
                     label=rf'WR Fitted, $\bar{{t_{{c,WR}}}}={ana_model.avg_conflict_duration_wr:.1f}$s')
    conf_axs[4].set_ylabel('Total no. of los [-]')
    conf_axs[4].plot(ana_model.n_inst, ana_model.los_total_nr, color='lightblue', linestyle='--',
                     label=f'NR Fitted, False conflicts={ana_model.false_conflict_ratio * 100:.0f}%')
    conf_axs[4].plot(ana_model.n_inst, ana_model.los_total_wr, color='coral', linestyle='--',
                     label=f'WR Fitted, Resolved={ana_model.resolve_ratio * 100:.0f}%')
    conf_axs[5].plot(ana_model.n_inst, ana_model.n_total, color='purple', label='NR/WR Model')
    conf_axs[5].set_ylabel('Total no. of A/C [-]')

    for ax in conf_axs:
        ax.set_xlabel('NR Inst. no. of aircraft [-]')
        # ax.set_ylim(conf_ylim[ax])
        ax.legend()

    # flst_ylim = {ax: ax.get_ylim() for ax in flst_axs}
    flst_axs[0].set_ylabel('Mean flight time [s]')
    flst_axs[0].plot(ana_model.n_inst, np.ones(ana_model.n_inst.shape) * ana_model.avg_duration,
                     color='blue', label='NR Model')
    flst_axs[0].plot(ana_model.n_inst, ana_model.mean_duration_wr, color='red', label='WR Model')
    flst_axs[1].set_ylabel('Mean 2D distance [m]')
    flst_axs[1].plot(ana_model.n_inst, np.ones(ana_model.n_inst.shape) * ana_model.urban_grid.avg_route_length,
                     color='purple', label='NR/WR Model')
    flst_axs[2].set_ylabel('Mean velocity [m/s]')
    flst_axs[2].plot(ana_model.n_inst, np.ones(ana_model.n_inst.shape) * ana_model.speed,
                     color='blue', label='NR Model')
    flst_axs[2].plot(ana_model.n_inst, ana_model.mean_v_wr, color='red', label='WR Model')
    for ax in flst_axs:
        ax.set_xlabel('NR Inst. no. of aircraft [-]')
        # ax.set_ylim(flst_ylim[ax])
        ax.legend()

    return [conf_fig, flst_fig], data


def save_figures(fig_list: List[plt.figure], name: str, output_dir: Path = OUTPUT_FOLDER) -> None:
    """
    Saves the figures

    :param fig_list: List with conf_fig and flst_fig
    :param name: save name
    :param output_dir:
    :return: None
    """
    print('Saving figures...')
    for fig in fig_list:
        fig.set_size_inches((16, 8), forward=False)
        fig.savefig(output_dir / 'RESULT' / f'{name}_{fig.number}.svg', bbox_inches='tight')


def save_data(data: dict, name: str, output_dir: Path = OUTPUT_FOLDER) -> pd.DataFrame:
    """
    Saves the data to a csv.

    :param data: data dict
    :param name: save name
    :param output_dir:
    :return: data dataframe
    """
    df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([data.keys(), data['NR'].keys()], names=['Reso', 'Parameter'])
    )
    for reso in data.keys():
        for param in data[reso].keys():
            df[reso, param] = data[reso][param]
    df.to_csv(output_dir / 'RESULT' / f'{name}.csv')
    return df


if __name__ == '__main__':
    use_pkl = True

    if use_pkl:
        res_pkl = Path(r'C:\Users\michi\OneDrive\Documenten\GitHub\bluesky\output\RESULT\batch_extensive_run_NR.pkl')
        with open(res_pkl, 'rb') as f:
            res = pkl.load(f)
    else:
        res = create_result_dict()
        res = process_result(res)
        save_result(res)

    grid, analytical = load_analytical_model(res)
    figs, data_dict = plot_result(res, analytical)
    save_figures(figs, res['name'])
    data_df = save_data(data_dict, res['name'])
