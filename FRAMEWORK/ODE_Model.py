# -*- coding: utf-8 -*-
"""ODE model layer

This module contains the ODE model implementation for inclusion of ODE models into tensorflow model

"""

# Importing dependencies
import tensorflow as tf
from tensorflow.keras import Model
from System import OdeSettings
from typing import List
from tensorflow.keras.layers import Input, Lambda
import tensorflow_probability as tfp
import logging
import numpy as np
logger = logging.getLogger(__name__)


class ODEModelLayer(tf.keras.layers.Layer):
    """ODE model layer

    Attributes:
        system: System object

    """

    def __init__(self, ode_settings: OdeSettings,
                 variable_dimensions, constitutive_model: Model = None):
        super(ODEModelLayer, self).__init__()
        """
        Creating tensorflow model object

        Args:
            ode_settings: System object with model specifications
            constitutive_model: Data-driven constitutive model (for model predictions)
            variable_dimensions: list of dictionaries containing the y, x and z dimensions and id's
        """
        self.ode_settings = ode_settings
        self.constitutive_model = constitutive_model
        self.variable_dimensions = variable_dimensions
        self.variable_splits = [np.prod(variable_details['dims']) for variable_details in self.variable_dimensions['x']]
        #ODE model variables
        # self.yield_x_glc = tf.keras.backend.variable(0.2, dtype=tf.float32,name ="Yield x_glc")
        # self.yield_x_gln = tf.keras.backend.variable(2, dtype=tf.float32,name ="Yield x_gln")
        # self.yield_x_glu = tf.keras.backend.variable(0.2, dtype=tf.float32,name ="Yield x_glu")
        # self.yield_glc_lac = tf.keras.backend.variable(1, dtype=tf.float32,name ="Yield glc_lac")
        # self.yield_glc_product = tf.keras.backend.variable(0.4, dtype=tf.float32,name ="Yield glc_product")
        # self.yield_gln_product = tf.keras.backend.variable(1, dtype=tf.float32,name ="Yield gln_product")
        # self.yield_glu_product = tf.keras.backend.variable(2, dtype=tf.float32,name ="Yield glu_product")
        # self.yield_glc_osmo = tf.keras.backend.variable(0.1, dtype=tf.float32,name ="Yield glc_osmo")
        # self.yield_gln_osmo = tf.keras.backend.variable(0.1, dtype=tf.float32,name ="Yield gln_osmo")
        # self.yield_lac_osmo = tf.keras.backend.variable(0.1, dtype=tf.float32,name ="Yield lac_osmo")
        # self.yield_glu_osmo = tf.keras.backend.variable(0.5, dtype=tf.float32,name ="Yield glu_osmo")
        # self.yield_nh4_osmo = tf.keras.backend.variable(0.5, dtype=tf.float32,name ="Yield nh4_osmo")
        # self.yield_gln_nh4 = tf.keras.backend.variable(0.2, dtype=tf.float32,name ="Yield gln_nh4")
        # self.yield_glc_nh4 = tf.keras.backend.variable(0.3, dtype=tf.float32,name ="Yield glc_nh4")
        # self.yield_gln_glu = tf.keras.backend.variable(0.5, dtype=tf.float32,name ="Yield gln_glu")


    def restore_variable_structure_batch(self, x_flat: tf.Tensor) -> List[tf.Tensor]:
        batch_size = tf.shape(x_flat)[0]
        x_split = tf.split(x_flat, num_or_size_splits=self.variable_splits, axis=-1)
        x_reshape = [tf.reshape(x_split[variable_id], shape=[batch_size]+variable_detail['dims'])
                     for variable_id, variable_detail in enumerate(self.variable_dimensions['x'])]
        return x_reshape

    def restore_variable_structure(self, x_flat: tf.Tensor) -> List[tf.Tensor]:
        x_split = tf.split(x_flat, num_or_size_splits=self.variable_splits, axis=-1)
        x_reshape = [tf.reshape(x_split[variable_id], shape=variable_detail['dims'])
                     for variable_id, variable_detail in enumerate(self.variable_dimensions['x'])]
        return x_reshape

    def flatten_variables_batch(self, x_list: List[tf.Tensor]) -> tf.Tensor:
        batch_size = tf.shape(x_list[0])[0]
        x_flatten = tf.concat([tf.reshape(x, shape=[batch_size, self.variable_splits[x_id]])
                               for x_id, x in enumerate(x_list)], axis=-1)
        return x_flatten

    def flatten_variables(self, x_list: List[tf.Tensor]) -> tf.Tensor:
        x_flatten = tf.concat([tf.reshape(x, shape=[self.variable_splits[x_id]])
                               for x_id, x in enumerate(x_list)], axis=-1)
        return x_flatten

    def batch(self, tensors: List[tf.Tensor]) -> tf.Tensor:
        """
        Call ODE solver for batch data

        Args:
            tensors: List of tensors: List of tensor, [0]: initial conditions (X(t0)=x0),
                                                      [1]: model constants (z),
                                                      [2]: time horizon (delta t),
                                                      [3]: constitutive_model predictions (y)
        Returns:
            Tensor: Tensor of future state variables (X(t0+delta t))
        """
        # Get batch size
        batch_size = tf.shape(tensors[0][0])[0]
        # Define loop function
        fun = lambda train_index: self.solve_ode(train_index, tensors)
        # Define batch elements to loop over
        train_indexes = tf.range(batch_size)
        # Run loop for batch size
        x_out = tf.map_fn(fun, train_indexes, dtype=tf.float32)
        x_out = tf.expand_dims(x_out, axis=1)
        return x_out

    def solve_ode(self, train_index: tf.Tensor, tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        # Unpack and define the training example data
        initial_state = self.flatten_variables_batch(tensors[2])
        initial_state = initial_state[train_index, ...] # This is the initial concentration, cond, q
        model_constants = [tensor[train_index, ...] for tensor in tensors[1]]  # This is the v
        delta_t = tensors[3][train_index]  # Time horizon
        constitutive_model_predictions = [tensor[train_index, ...] for tensor in tensors[0]] #This is the K

        solver = tfp.math.ode.DormandPrince(rtol=self.ode_settings.rel_tol,
                                            atol=self.ode_settings.abs_tol)
        # Solve ODE
        x1 = solver.solve(self.ode,
                          initial_time=0,
                          initial_state=initial_state,
                          solution_times=tfp.math.ode.ChosenBySolver(tf.squeeze(delta_t)),  # [delta_t],
                          constants={
                              'constitutive_model_predictions': constitutive_model_predictions,
                              'model_constants': model_constants})
        # Extract solution for t=t+dt
        x1 = tf.expand_dims(x1.states[-1, :], axis=0)  # [-1, :]
        return x1


    def call(self, tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Run solver for ODE model for a single training example with index train_index

        Args:
            List of tensors: List of tensor, [0]: initial conditions (X(t0)=x0),
                                             [1]: time horizon (delta t)
                                             [2]: constitutive_model predictions (y),
                                             [3]: model constants (z),
        Returns:
            tensor: Tensor of future state variables (X(t0+delta t))
        """
        # # Get batch size
        # batch_size = tf.shape(tensors[0][0])[0]
        #
        # # Unpack all inputs
        # initial_states = tensors[2]
        # delta_t = tf.squeeze(tensors[3], axis=-1)
        # constitutive_model_predictions = tensors[0]
        # model_constants = tensors[1]
        #
        # # Flatten input for ODE solver
        # x0 = self.flatten_variables(initial_states)
        #
        # # Solve ODE
        # ode_constants = {'constitutive_model_predictions': constitutive_model_predictions,
        #                  'model_constants': model_constants}
        #
        # dt_index_1 = tf.argsort(delta_t, axis=-1)
        # dt_index_2 = tf.argsort(dt_index_1, axis=-1)
        # dt = tf.gather(delta_t, dt_index_1)
        #
        # x1 = tfp.math.ode.DormandPrince(rtol=self.ode_settings.rel_tol,
        #                                 atol=self.ode_settings.abs_tol).solve(self.ode,
        #                                                                       initial_time=0,
        #                                                                       initial_state=x0,
        #                                                                       solution_times=dt,
        #                                                                       constants=ode_constants)
        #
        # # Extract solution for t=t+dt and reshape
        # fun = lambda train_index: x1.states[dt_index_2[train_index]][train_index, :]
        # x_1_collected = tf.map_fn(fun, tf.range(batch_size), dtype=tf.float32)
        x_1_collected = self.batch(tensors=tensors)
        x_1_out = self.restore_variable_structure_batch(x_1_collected)
        return x_1_out

    def ode(self, t: tf.Tensor, x: tf.Tensor, constitutive_model_predictions: List[tf.Tensor],
            model_constants: List[tf.Tensor]) -> tf.Tensor:
        """
        Calculate RHS of system of ODEs

        Args:
            x (tensor): State tensor: state variables at t
            t (tensor): Time tensor
            constitutive_model_predictions (list of tensors): List of constitutive model predictions : rates predicted by ANN model
            model_constants (list of tensors): List of constants for model (z): model constants for ODEs
        Returns:
            tensor: dxdt tensor with time-derivative of all state variables x at t + delta t
        """
        # Reshape x-variable (add this as the first line in the ODE-definition)
        x = self.restore_variable_structure(x)

        if self.constitutive_model is not None:
            constitutive_model_predictions = self.constitutive_model(x)

        rGlc = constitutive_model_predictions[0]*x[0]*x[1]
        rGln = constitutive_model_predictions[1]*x[0]*x[2]
        rLac = constitutive_model_predictions[2]*x[0]*x[3]
        rGlu = constitutive_model_predictions[3]*x[0]*x[4]
        rNH4 = constitutive_model_predictions[4]*x[0]*x[5]
        rOsmo = constitutive_model_predictions[5]*x[0]*x[7]

        kD = constitutive_model_predictions[6]*x[0]
        rDegPRODUCT = constitutive_model_predictions[7]*x[6]

        rate_glc_to_x = rGlc*constitutive_model_predictions[8]
        rate_gln_to_x = rGln*constitutive_model_predictions[9]
        rate_glu_to_x = rGlu*constitutive_model_predictions[10]
        rate_glc_to_product = rGlc*constitutive_model_predictions[11]
        rate_gln_to_product = rGln*constitutive_model_predictions[12]
        rate_glu_to_product = rGlu*constitutive_model_predictions[13]
        rate_glc_to_lac = rGlc*constitutive_model_predictions[14]
        rate_lac_to_osmo = x[1] * constitutive_model_predictions[15]+ \
                           x[2] * constitutive_model_predictions[16]+ \
                           x[3] * constitutive_model_predictions[17]+ \
                           x[4] * constitutive_model_predictions[18]+ \
                           x[5] * constitutive_model_predictions[19]
        rate_gln_to_nh4 = rGln*constitutive_model_predictions[20]
        rate_glc_to_nh4 = rGlc*constitutive_model_predictions[21]
        rate_gln_to_glu = rGln*constitutive_model_predictions[22]
        #
        # rate_glc_to_x = rGlc*tf.abs(self.yield_x_glc)
        # rate_gln_to_x = rGln*tf.abs(self.yield_x_gln)
        # rate_glu_to_x = rGlu*tf.abs(self.yield_x_glu)
        # rate_glc_to_product = rGlc*tf.abs(self.yield_glc_product)
        # rate_gln_to_product = rGln*tf.abs(self.yield_gln_product)
        # rate_glu_to_product = rGlu*tf.abs(self.yield_glu_product)
        # rate_glc_to_lac = rGlc*tf.abs(self.yield_glc_lac)
        # rate_lac_to_osmo = x[1] * tf.abs(self.yield_glc_osmo)+ \
        #                    x[2] * tf.abs(self.yield_gln_osmo)+ \
        #                    x[3] * tf.abs(self.yield_lac_osmo)+ \
        #                    x[4] * tf.abs(self.yield_glu_osmo)+ \
        #                    x[5] * tf.abs(self.yield_nh4_osmo)
        # rate_gln_to_nh4 = rGln*tf.abs(self.yield_gln_nh4)
        # rate_glc_to_nh4 = rGlc*tf.abs(self.yield_glc_nh4)
        # rate_gln_to_glu = rGln*tf.abs(self.yield_gln_glu)


        dBiomassdt =  rate_glc_to_x + rate_gln_to_x + rate_glu_to_x-kD\
                      -(model_constants[1] / model_constants[0]) * x[0]

        dGlcdt = (model_constants[3] / model_constants[0]) * (model_constants[4] - x[1]) - rGlc
        dGlndt = (model_constants[3] / model_constants[0]) * (model_constants[5] - x[2]) - rGln

        dLacdt = -((model_constants[3] / model_constants[0]) * x[3]) + rate_glc_to_lac - rLac

        dGludt = (model_constants[3] / model_constants[0]) * (model_constants[6] - x[4])+rate_gln_to_glu - rGlu

        dNH4dt = (model_constants[3] / model_constants[0]) * (model_constants[7] - x[5]) + rate_gln_to_nh4 + rate_glc_to_nh4 \
                  - rNH4

        dPRODUCTdt = -(model_constants[3] / model_constants[0]) * x[6] + rate_glc_to_product+\
                 rate_gln_to_product + rate_glu_to_product -rDegPRODUCT

        dOsmolalitydt = (model_constants[3] / model_constants[0]) * (model_constants[8]-x[7])-rOsmo\
                        +rate_lac_to_osmo



        dOffline_ph = tf.zeros_like(x[7], dtype=tf.float32)

        dO2 = tf.zeros_like(x[8], dtype=tf.float32)


        dxdt = [dBiomassdt,
                  dGlcdt,
                  dGlndt,
                  dLacdt,
                  dGludt,
                  dNH4dt,
                  dPRODUCTdt,
                  dOsmolalitydt,
                  dOffline_ph,
                  dO2]

        # Flatten output (add this as the last line in the ODE-definition)
        dxdt = self.flatten_variables(dxdt)
        return dxdt


class ODEModel:
    def __init__(self, ode_settings: OdeSettings,
                 variable_dimensions, constitutive_model: Model = None):
        self.ODEModelLayer = ODEModelLayer(ode_settings=ode_settings, variable_dimensions=variable_dimensions,
                                           constitutive_model=constitutive_model)
        self.ode_settings = ode_settings
        self.variable_dimensions = variable_dimensions
        self.constitutive_model = constitutive_model

    def create_ODE_model(self, name) -> Model:

        # Set up the initial conditions based on x_dimensions
        initial_conditions = [Input(shape=tuple(self.variable_dimensions['x'][i]['dims']), name=str(self.variable_dimensions['x'][i]['id'])+'_initial')
                              for i in range(len(self.variable_dimensions['x']))]

        # Set up the constitutive_model_predictions based on y dimensions
        constitutive_model_predictions = [Input(shape=tuple(self.variable_dimensions['y'][i]['dims']), name=str(self.variable_dimensions['y'][i]['id']))
                              for i in range(len(self.variable_dimensions['y']))]
        # Set up the constants based on z dimensions
        constants = [Input(shape=tuple(self.variable_dimensions['z'][i]['dims']), name=str(self.variable_dimensions['z'][i]['id']))
                              for i in range(len(self.variable_dimensions['z']))]

        # Set the time horizon
        delta_t = Input(shape=(1,), name='Delta_t')

        # Set up layer
        x1 = self.ODEModelLayer([constitutive_model_predictions,constants,initial_conditions, delta_t])

        # Generate PBM model
        ODE_model = Model(inputs=[constitutive_model_predictions,constants,initial_conditions,delta_t],
                          outputs=x1)

        ODE_model.summary(print_fn=logger.debug)
        return ODE_model
