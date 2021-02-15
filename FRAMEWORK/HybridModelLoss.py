# -*- coding: utf-8 -*-
"""Hybrid model loss module

This module contains a customized model loss for training of hybrid model

"""

# Importing dependencies
import tensorflow as tf
from System import System
from tensorflow import keras




class HybridModelLoss:
    """HybridModelLoss class

    Attributes:
        system: Model system settings
    """

    def __init__(self, system: System):
        """
        Creating loss object

        Args:
            system (object): System specifications
        """
        self.system = system

    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate model loss

        Args:
            y_true: Experimentally measured variables
            y_pred: Predicted future variables
        Returns:
            tensor: Tensor object with mean absolute error loss
        """

        loss_value = tf.abs(y_pred - y_true)


        return loss_value
