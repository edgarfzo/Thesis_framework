# -*- coding: utf-8 -*-
"""Time series pair module

This module converts a data-object to pairs of time-series data points, used for training hybrid model

"""

# Importing dependencies


import numpy as np
from System import System
from typing import List, Dict
from Data import Data





class TimeSeriesPair:
    """Class containing all data

    Attributes:
        data: Data object
        system: System object
    """

    def __init__(self, data: Data, system: System) -> None:
        """
        Creating training data object

        Args:
            data: Data object
            system: System object
        """
        self.data = data
        self.system = system

    def shuffle(self, pool_type: List[str], start: int = 0, stop: int = np.inf, min_step: int = 1,
                max_step: int = np.inf, ) -> Dict[str, np.ndarray]:
        """
        Shuffle data points

        Args:
            pool_type: Pool type
            start: Start index for shuffling
            stop: Stop index for shuffling
            min_step: Minimum step in shuffling
            max_step: Maximum step in shuffling
        Returns:
            Dictionary of training data
        """

        # Initialize output
        batch_description = []
        measured_variable = []
        future_variable = []
        time_horizon = []
        # Shuffle input data
        for batch in self.data.batches:
            for pool in pool_type:
                if pool in batch.pool and batch.pool[pool]:
                    # Number of measurement
                    number_of_measurements = len(batch.measurements)
                    # Define shuffle parameters
                    shuffle_end = min(number_of_measurements-1, stop)
                    shuffle_start = max(0, min(start, shuffle_end - min_step))
                    for start_index in range(shuffle_start, shuffle_end):
                        for end_index in range(start_index + min_step,
                                               min(start_index + max_step, shuffle_end)+1):
                            # Load measurements for start and end indexes
                            start_measurement = batch.measurements[start_index]
                            future_measurement = batch.measurements[end_index]
                            # Measured and future variables
                            batch_info = []
                            measured = []
                            future = []
                            delta_time =[]

                            for sensor in start_measurement.external_sensors[:3]:
                                batch_info.append(str(int(sensor.value)))
                            batch_info.append(str(float(start_measurement.external_sensors[4].value)))
                            for sensor in start_measurement.external_sensors[4:]:
                                measured.append(float(sensor.value))
                            for sensor in future_measurement.external_sensors[4:]:
                                future.append(float(sensor.value))
                            delta_time.append(future_measurement.time-start_measurement.time)

                            # Save variables
                            batch_description.append(batch_info)
                            measured_variable.append(measured)
                            future_variable.append(future)
                            time_horizon.append(delta_time)

        time_horizon = np.asarray(time_horizon)

        return {'Batch info': batch_description,
            'Measured variables': np.asarray(measured_variable),
                'Future variables': np.asarray(future_variable),
                'Time horizon': time_horizon}

