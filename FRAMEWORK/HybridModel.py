# -*- coding: utf-8 -*-
"""Hybrid model module

This module contains the model structure generator for the hybrid particle model

"""
# Importing dependencies

import logging
from ODE_Model import ODEModel
from Rates_Model import RateModel
from HybridModelLoss import HybridModelLoss
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from System import System
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import pandas as pd

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

class HybridModel:
    def __init__(self, system: System) -> None:

        # Create sub-models dictionary and model
        self.sub_models = {}
        self.training_model: Model
        self.system = system


        # Get phenomena rate sizes
        # self.rate_sizes = self.rate_sizes(system)

        # Initialize loss model
        self.loss_model = None

        self.variable_details = {'y': [{'dims': [1],'id': 'k0'},
                                       {'dims': [1],'id': 'k1'},
                                       {'dims': [1],'id': 'k2'},
                                       {'dims': [1],'id': 'k3'},
                                       {'dims': [1],'id': 'k4'},
                                       {'dims': [1],'id': 'k5'},
                                       {'dims': [1],'id': 'k6'},
                                       {'dims': [1],'id': 'k7'},
                                       {'dims': [1],'id': 'k8'},
                                       {'dims': [1],'id': 'k9'},
                                       {'dims': [1],'id': 'k10'},
                                       {'dims': [1],'id': 'k11'},
                                       {'dims': [1],'id': 'k12'},
                                       {'dims': [1],'id': 'k13'},
                                       {'dims': [1],'id': 'k14'},
                                       {'dims': [1],'id': 'k15'},
                                       {'dims': [1],'id': 'k16'},
                                       {'dims': [1],'id': 'k17'},
                                       {'dims': [1],'id': 'k18'},
                                       {'dims': [1],'id': 'k19'},
                                       {'dims': [1],'id': 'k20'},
                                       {'dims': [1],'id': 'k21'},
                                       {'dims': [1],'id': 'k22'},








                                      ],
                                'z': [{'dims': [1],'id': 'Volume'},
                                       {'dims': [1],'id': 'Bleed_rate'},
                                       {'dims': [1],'id': 'Harvest_rate'},
                                       {'dims': [1],'id': 'Dilution'},
                                       {'dims': [1],'id': 'Glc_feed'},
                                       {'dims': [1],'id': 'Gln_feed'},
                                       {'dims': [1],'id': 'Glu_feed'},
                                       {'dims': [1],'id': 'Ammonium_feed'},
                                       {'dims': [1],'id': 'Osmolality_feed'}
                                      ],
                                 'x': [{'dims': [1],'id': 'Biomass'},
                                       {'dims': [1],'id': 'Glucose'},
                                       {'dims': [1],'id': 'Glutamine'},
                                       {'dims': [1],'id': 'Lactose'},
                                       {'dims': [1],'id': 'Glutamate'},
                                       {'dims': [1],'id': 'Ammonia'},
                                       {'dims': [1],'id': 'PRODUCT'},
                                       {'dims': [1],'id': 'Osmolality'},
                                       {'dims': [1],'id': 'pH'},
                                       {'dims': [1],'id': 'Po2'}]}

        # Set up hybrid sub-models
        self.create_sub_models()

        # Set up overall hybrid model
        self.create_hybrid_model()
        self.ref_loss_train = None
        self.ref_loss_val = None
        self.training_history = None


    def create_sub_models(self):
         #Creation of ANN/RATE model

        self.sub_models['rate'] = RateModel(rate_settings = self.system.rate_settings,variable_dimensions = self.variable_details).create_Rate_model(name='Ratemodel')

        self.sub_models['rate'].summary()
        print('RATE MODEL SET UP')

        #Creation of ODE model
        self.sub_models['ODE'] = ODEModel(ode_settings = self.system.ode_settings,
                                          variable_dimensions = self.variable_details).create_ODE_model(name='ODEmodel')
        self.sub_models['ODE'].summary()

        print('ODE MODEL SET UP')

        #Creation of ODE model for multi-step predictions
        self.sub_models['ODE_pred'] = ODEModel(ode_settings = self.system.ode_settings,
                                          variable_dimensions = self.variable_details,
                                               constitutive_model = self.sub_models['rate']).create_ODE_model(name='ODEpredmodel')
        self.sub_models['ODE_pred'].summary()

        print('ODE MODEL_PRED SET UP')


        #Creation of loss model

        self.loss_model = HybridModelLoss(system=self.system)


    def create_hybrid_model(self):
        # Define hybrid model inputs and input size
        measured_constants = [Input(shape=tuple(self.variable_details['z'][i]['dims']), name=str(self.variable_details['z'][i]['id']))
                              for i in range(len(self.variable_details['z']))]

        initial_conditions = [Input(shape=tuple(self.variable_details['x'][i]['dims']), name=str(self.variable_details['x'][i]['id'])+'_initial')
                              for i in range(len(self.variable_details['x']))]

        time = Input(shape=(1,), name='Time')

        # Model structure definition

        rates = self.sub_models['rate']([initial_conditions])

        variable_prediction = self.sub_models['ODE']([rates,measured_constants,initial_conditions,time])
        #output = tf.keras.layers.Concatenate(axis = -1)(variable_prediction)
        # Define training model
        self.training_model = Model(inputs=[measured_constants,initial_conditions,time],
                                    outputs=variable_prediction)

        self.training_model.summary()
        print('HYBRID MODEL SET UP')



    def calculate_reference_loss(self, x, y, category: str):
        y_tensor = tf.constant(y, dtype=tf.float32)
        x_tensor = tf.constant(x, dtype=tf.float32)
        reference_loss = self.loss_model.loss(y_tensor, x_tensor).numpy()
        av_reference_loss = np.mean(reference_loss)
        if category == 'train':
            self.ref_loss_train = {'Reference loss': reference_loss,
                                   'Average loss': av_reference_loss}
            #print('Training reference error: ' + str(self.ref_loss_train['Average loss']))
            return self.ref_loss_train['Average loss']
        elif category == 'val':
            self.ref_loss_val = {'Reference loss': reference_loss,
                                 'Average loss': av_reference_loss}
            #print('Validation reference error: ' + str(self.ref_loss_val['Average loss']))
            return self.ref_loss_val['Average loss']
        else:
            print('category not properly set')



    @staticmethod
    def model_data(shuffled_data,included_variables):


        measured_constants = [np.expand_dims(shuffled_data['Measured variables'][:,i],axis = 1)
                              for i in range(len(included_variables),len(included_variables)+9)]  # 8 constants
        initial_conditions = [np.expand_dims(shuffled_data['Measured variables'][:,i],axis = 1) for i in range(len(included_variables))]  # 14 initial conditions
        time = np.expand_dims(np.random.normal(loc=shuffled_data['Time horizon'], scale=0.0001), axis = 1)  # 1 time horizon

        y_future_variables = [np.expand_dims(shuffled_data['Future variables'][:,i],axis = 1) for i in range(len(included_variables))]  # 14 initial conditions

        x = [measured_constants,initial_conditions,time]

        return [x, y_future_variables]







