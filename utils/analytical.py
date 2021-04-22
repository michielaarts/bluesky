"""
The analytical model base class.

Created by Michiel Aarts, April 2021
"""
import pandas as pd
from typing import Tuple
import numpy as np


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

        # NR model.
        self.c_inst_nr = None
        self.los_inst_nr = None

        # WR delay model.
        self.delays = None
        self.mean_v_wr = None
        self.n_inst_wr = None
        self.mean_flight_time_nr = None
        self.mean_flight_time_wr = None
        self.flow_rate_wr = None

        # Fitted variables.
        self.c_inst_wr_fitted = None
        self.mean_conflict_duration_nr = None  # s
        self.mean_conflict_duration_wr = None  # s
        self.c_total_nr = None
        self.c_total_wr = None
        self.c_total_wr_fitted = None
        self.false_conflict_ratio = None
        self.resolve_ratio = None
        self.los_total_nr = None
        self.los_total_wr = None
        self.mean_los_duration_nr = None  # s
        self.mean_los_duration_wr = None  # s
        self.los_inst_nr_fitted = None
        self.los_inst_wr = None

    def nr_model(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def delay_model(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        pass

    def wr_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def fit_derivatives(self, data: dict) -> None:
        pass

    def wr_conflict_model(self):
        pass
