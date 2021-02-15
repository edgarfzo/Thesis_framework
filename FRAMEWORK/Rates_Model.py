# -*- coding: utf-8 -*-
"""Rate model layer

This module contains the ODE model implementation for inclusion of ODE models into tensorflow model

"""

# Importing dependencies
import logging


from tensorflow.keras.layers import  Reshape, Concatenate
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda
import logging
from System import RateSettings
import numpy as np
logger = logging.getLogger(__name__)

class RateModel:
    """Rate model layer

    Attributes:
        system: System object
    """

    def __init__(self, variable_dimensions,rate_settings : RateSettings):
        """
        Creating tensorflow model object

        Args:
            ode_settings: System object with model specifications
            constitutive_model: Data-driven constitutive model (for model predictions)
            variable_dimensions: list of dictionaries containing the y, x and z dimensions and id's
        """
        self.variable_dimensions = variable_dimensions
        self.variable_splits = [np.prod(variable_details['dims']) for variable_details in self.variable_dimensions['x']]
        self.rate_settings = rate_settings

    def create_Rate_model(self, name) -> Model:

        # Set up the initial conditions based on x_dimensions
        inputs = [Input(shape = tuple(self.variable_dimensions['x'][i]['dims']),name = str(self.variable_dimensions['x'][i]['id']))
            for i in range(len(self.variable_dimensions['x']))]

        variable_splits = [np.prod(variable_detail['dims']) for variable_detail in self.variable_dimensions['x']]
        x_flatten = Concatenate(axis = -1)([Reshape((variable_splits[x_id],))(x) for x_id,x in enumerate(inputs)])
        x_flatten = layers.BatchNormalization()(x_flatten)
        x = layers.Dense(self.rate_settings.layer_neurons[0], activation = self.rate_settings.layer_activations[0])(x_flatten)
        x = layers.Dense(self.rate_settings.layer_neurons[1], activation = self.rate_settings.layer_activations[1])(x)
        #x = layers.Dense(self.rate_settings.layer_neurons[2], activation = self.rate_settings.layer_activations[2])(x)
        outputs = Lambda(
            lambda y: tf.split(tf.abs(y),[np.prod(variable['dims']) for variable in self.variable_dimensions['y']],axis = -1),
            name = 'Rate_splitter')(x)

        Rate_model = keras.Model(inputs = inputs,outputs = outputs)

        return Rate_model