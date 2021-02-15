# -*- coding: utf-8 -*-
"""Data structure

This module contains universal data structure objects for storage of time-series data

"""

# Importing dependencies
import pandas as pd
from typing import List
import datetime
import numpy as np



class Sensor:
    """Class for single sensor input

    Attributes:
        sensor_id: Name of sensor
        value: Measurement value
        data_type: Data type (single_value or distribution)
        std_error: Standard error of measurement

    """

    def __init__(self, sensor_id: str, value: [float, np.ndarray], data_type: str,
                 std_error: [float, np.ndarray] = None, unit: str = None) -> None:
        """
        Creating sensor object

        Args:
            sensor_id: Name of sensor
            value: Sensor value
            data_type: Type of data
            std_error: Standard error of measurement
        """
        self.sensor_id = sensor_id
        self.value = value
        self.data_type = data_type
        self.std_error = std_error
        self.unit = unit


class Measurement:
    """Class for single measurement

    Attributes:
        measurement_id: Measurement name
        external_sensors: List of external sensor inputs
        particle_analysis_sensors: List of particle analysis sensor inputs
        time: Time of measurement in datetime format
    """

    def __init__(self, measurement_id: str, time: datetime.datetime) -> None:
        """
        Creating measurement object

        Args:
            measurement_id: Name of measurement
            time: Time of measurement (datetime object)
        """
        self.measurement_id = measurement_id
        self.external_sensors: List[Sensor] = []
        self.particle_analysis_sensors: List[Sensor] = []
        self.image_sensors: List[Sensor] = []
        self.time = time

    def add_external_sensor(self, sensor: Sensor) -> None:
        """
        Add sensor object to list of external sensors

        Args:
            sensor: Sensor object
        """
        self.external_sensors.append(sensor)

    def add_particle_analysis_sensor(self, sensor: Sensor) -> None:
        """
        Add sensor object to list of particle analysis sensors

        Args:
            sensor: Sensor object
        """
        self.particle_analysis_sensors.append(sensor)

    def add_image_sensor(self, sensor: Sensor) -> None:
        """
        Add sensor object to list of image sensors

        Args:
            sensor: Sensor object
        """
        self.image_sensors.append(sensor)


class Batch:
    """Class for batch data

    Attributes:
        batch_id: Name of batch (use unique naming)
        measurements: List of measurement objects
        pool: Dictionary of data pool types
    """

    def __init__(self, batch_id: str) -> None:
        """
        Creating batch object

        Args:
            batch_id: Name of batch
        """
        self.batch_id = batch_id
        self.measurements: List[Measurement] = []
        self.pool = dict()

    def add_measurement(self, measurement: Measurement) -> None:
        """
        Add measurement object to list of measurements

        Args:
            measurement: Measurement object
        """
        self.measurements.append(measurement)

    def add_measurment_from_text(self, time, temperature, density, density_temperature, image_analysis_data):
        measurement_object = Measurement(measurement_id='Manual measurement', time=time)
        # Get list of image analysis sensors
        list_of_image_analysis_sensors = image_analysis_data.columns.values.tolist()
        for image_analysis_sensor_id in list_of_image_analysis_sensors:
            value = image_analysis_data.loc[:, image_analysis_sensor_id].values
            particle_analysis_sensor_object = Sensor(image_analysis_sensor_id, value, 'image_analysis')
            measurement_object.add_particle_analysis_sensor(particle_analysis_sensor_object)
        # Add other sensors
        for sensor_reading, external_sensor_id in zip([temperature, density, density_temperature], ['Temperature','Density','Density_temperature']):
            value = sensor_reading
            external_sensor_object = Sensor(external_sensor_id, value, 'external')
            measurement_object.add_external_sensor(external_sensor_object)
        # Add concentration sensor
        conc_calibration = np.array([3.14794, -0.010415, 0.0201494,
                                     6.24689e-06, -0.000299322, 1.58896e-05])
        def concentration_model(coef, temp, dens):
            dens = dens*1000
            input_matrix = np.array([1,
                                     dens,
                                     temp,
                                     dens ** 2,
                                     temp ** 2,
                                     dens * temp]).T
            out = np.sum(coef * input_matrix)
            return out
        external_sensor_object = Sensor('Concentration', concentration_model(conc_calibration,
                                                                             density_temperature,
                                                                             density), 'external')
        measurement_object.add_external_sensor(external_sensor_object)
        self.add_measurement(measurement_object)


class Data:
    """Class containing all data

    Attributes:
        batches: List of batches
        case_id: Case name
    """

    def __init__(self, case_id: str) -> None:
        """
        Creating data object

        Args:
            case_id: Case name
        """
        self.batches: List[Batch] = []
        self.case_id = case_id

    def add_batch(self, batch: Batch) -> None:
        """
        Add batch object to list of batches

        Args:
            batch: Batch object
        """
        self.batches.append(batch)

    def load_from_excel(self, rawdata,excluded_variables,included_variables) -> None:


        dataframe = pd.read_excel(rawdata)
        dataframe = dataframe.drop(columns=excluded_variables)
        stats = pd.DataFrame(columns = included_variables)
        mean = pd.DataFrame(columns = included_variables)
        stdev = pd.DataFrame(columns = included_variables)
        range_var = pd.DataFrame(columns = included_variables)
        for i in included_variables:
            stats[i] = dataframe[i].describe()
        for i in range(28):
            for j in included_variables:
                mean.loc[i,j] = dataframe[dataframe.Time_Sampling_Day == i][j].describe().loc["mean"]
                stdev.loc[i,j] = dataframe[dataframe.Time_Sampling_Day == i][j].describe().loc["std"]
                range_var.loc[i,j] = dataframe[dataframe.Time_Sampling_Day == i][j].describe().loc["max"]\
                            -dataframe[dataframe.Time_Sampling_Day == i][j].describe().loc["min"]


        for b in list(dataframe.Batch.unique()):
            batch = Batch(batch_id = b)
            df1 = dataframe[dataframe["Batch"]==b]
            df1["Glucose_feed"] = df1.Glucose.iloc[0]
            df1["Glutamine_feed"] = df1.Glutamine.iloc[0]
            df1["Glutamate_feed"] = df1.Glutamate.iloc[0]
            df1["Ammonium_feed"] = df1.Ammonium.iloc[0]
            df1["Osmolality_feed"] = df1.Osmolality.iloc[0]
            for t in list(df1.Time_Sampling_Day.unique()):
                measurement = Measurement(measurement_id = t,
                                          time = list(df1[df1.Time_Sampling_Day == t].Timepoint)[0])
                for p in list(df1.columns):
                    df2 = df1
                    if p in included_variables:
                        # breakpoint()
                        if t == 0:
                            df2[p].iloc[int(t)] = df1[p].iloc[int(t)]
                        elif t == 27:
                            df2[p].iloc[int(t)] = (df1[p].iloc[int(t)-1]+df1[p].iloc[int(t)]) / 2
                        elif t == 47:
                            df2[p].iloc[int(t)] = (df1[p].iloc[int(t)-1]+df1[p].iloc[int(t)]) / 2
                        else:

                            df2[p].iloc[int(t)] = (df1[p].iloc[int(t)-1]+df1[p].iloc[int(t)]+df1[p].iloc[int(t)+1]) / 3
                    else:
                        df2[p] = df1[p]

                    sensor = Sensor(sensor_id = p,
                                    value = df2[(df2.Time_Sampling_Day == t)][p],
                                    data_type = 'single_value',
                                    std_error = 0)

                    measurement.add_external_sensor(sensor)
                batch.add_measurement(measurement)
            self.add_batch(batch)
        return stats,mean,stdev,range_var


    def set_batch_pool(self, pool_batch_id: List[str], pool_type: str) -> None:
        """
        Set batch_id to data pool (overrides previous pool_type)

        Args:
            pool_batch_id: List of batch names to set in pool_type
            pool_type: Pool type (training, validation, test, etc.)
        """
        for batch_id in pool_batch_id:
            for batch in self.batches:
                if batch.batch_id == batch_id:
                    # Reset pool definition
                    batch.pool = dict()
                    batch.pool[pool_type] = True
                    print(str(batch_id) + " set to " + str(pool_type) + " data pool")
                    break

    def reset_batch_pools(self):
        for batch in self.batches:
            # Reset pool definition
            batch.pool = dict()
        print("Batch pool(s) have been reset")




