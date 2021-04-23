"""
The analytical model base class.

Created by Michiel Aarts, April 2021
"""
import pandas as pd
from typing import Tuple
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


class AnalyticalModel:
    def __init__(self):
        self.flow_ratio = None
        self.max_value = None
        self.accuracy = None
        self.duration = None
        self.speed = None
        self.s_h = None
        self.s_v = None
        self.t_l = None

        self.cruise_alt = None
        self.departure_alt = None
        self.mean_route_length = None

        # Base arrays.
        self.n_inst = None
        self.n_total = None

        # NR conflict model.
        self.c_inst_nr = None
        self.c_total_nr = None
        self.los_inst_nr = None
        self.los_total_nr = None

        # WR delay model.
        self.delays = None
        self.mean_v_wr = None
        self.n_inst_wr = None
        self.mean_flight_time_nr = None
        self.mean_flight_time_wr = None
        self.flow_rate_wr = None

        # WR conflict model.
        self.c_total_wr = None
        self.dep = None

        # Fitted variables.
        self.c_inst_wr_fitted = None
        self.mean_conflict_duration_nr = None  # s
        self.mean_conflict_duration_wr = None  # s
        self.c_total_nr_fitted = None
        self.c_total_wr_fitted = None
        self.false_conflict_ratio = None
        self.resolve_ratio = None
        self.los_total_nr_fitted = None
        self.los_total_wr_fitted = None
        self.mean_los_duration_nr = None  # s
        self.mean_los_duration_wr = None  # s
        self.los_inst_nr_fitted = None
        self.los_inst_wr_fitted = None

    def nr_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def delay_model(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        pass

    def wr_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def _fit_mean_duration(self, exp_inst: np.ndarray, exp_total: np.ndarray) -> float:
        """ Least squares fit of the mean conflict or los duration """
        if exp_inst.shape != exp_total.shape:
            raise ValueError('Inputs must be of same size')
        t_mean = opt.fmin(lambda a: np.nanmean(np.power(exp_inst * self.duration[1] / a - exp_total, 2)),
                          x0=2., disp=False)[0]
        return t_mean

    @staticmethod
    def _fit_conflict_los_ratio(exp_c_total: np.ndarray, exp_los_total: np.ndarray) -> float:
        """ Least squares fit of the false conflict ratio """
        if exp_c_total.shape != exp_los_total.shape:
            raise ValueError('Inputs must be of same size')
        ratio = opt.fmin(lambda a: np.nanmean(np.power(exp_c_total * a - exp_los_total, 2)),
                         x0=0.5, disp=False)[0]
        return 1 - ratio

    def _fit_c_inst_wr(self, exp_n_inst_nr: np.ndarray, exp_c_inst_wr: np.ndarray) -> np.ndarray:
        """ Least squares, second degree polynomial fit of the instantaneous number of conflicts with resolution """
        res = opt.fmin(lambda a: np.nanmean(np.power(a * exp_n_inst_nr * exp_n_inst_nr - exp_c_inst_wr, 2)),
                       x0=1, disp=False)[0]
        return res * np.power(self.n_inst, 2)

    def fit_derivatives(self, data: dict) -> None:
        """
        Fits all derivatives and functions of a given simulation experiment result.
        E.g.: mean conflict duration, false conflict percentage, etc.

        :param data: Data dictionary from utils/log_reader.py
        """
        print('Fitting derivatives...')
        # Set all unstable data to nan.
        for reso in data.keys():
            for key in data[reso].keys():
                if key != 'stable_filter':
                    data[reso][key] = np.where(data[reso]['stable_filter'], data[reso][key], np.nan)

        self.c_inst_wr_fitted = self._fit_c_inst_wr(data['NR']['ni_ac'], data['WR']['ni_conf'])
        self.mean_conflict_duration_nr = self._fit_mean_duration(data['NR']['ni_conf'], data['NR']['ntotal_conf'])
        self.mean_conflict_duration_wr = self._fit_mean_duration(data['WR']['ni_conf'], data['WR']['ntotal_conf'])
        self.mean_los_duration_nr = self._fit_mean_duration(data['NR']['ni_los'], data['NR']['ntotal_los'])
        self.mean_los_duration_wr = self._fit_mean_duration(data['WR']['ni_los'], data['WR']['ntotal_los'])
        self.false_conflict_ratio = self._fit_conflict_los_ratio(data['NR']['ntotal_conf'],
                                                                 data['NR']['ntotal_los'])
        self.resolve_ratio = self._fit_conflict_los_ratio(data['WR']['ntotal_conf'], data['WR']['ntotal_los'])

        self.c_total_nr_fitted = self.c_inst_nr * self.duration[1] / self.mean_conflict_duration_nr
        self.c_total_wr_fitted = self.c_inst_wr_fitted * self.duration[1] / self.mean_conflict_duration_wr
        self.los_total_nr_fitted = self.c_total_nr_fitted * (1 - self.false_conflict_ratio)
        self.los_total_wr_fitted = self.c_total_wr_fitted * (1 - self.resolve_ratio)
        self.los_inst_nr_fitted = self.los_total_nr_fitted * self.mean_los_duration_nr / self.duration[1]
        self.los_inst_wr_fitted = self.los_total_wr_fitted * self.mean_los_duration_wr / self.duration[1]

    def wr_conflict_model(self):
        pass

    def plot_mfd(self) -> None:
        """
        Plots the MFD of the provided analytical model.
        Useful when creating a scenario, to see whether the
        instantaneous numbers of aircraft without resolution
        are within the stable range.
        """
        plt.figure()
        plt.plot(self.n_inst, self.flow_rate_wr)
        plt.xlabel('No. of inst. aircraft NR')
        plt.ylabel('Flow rate [veh m / s]')
