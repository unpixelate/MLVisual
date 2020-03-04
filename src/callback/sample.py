from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import sys
from utils import helpers

class PredictionCallback(keras.callbacks.Callback):    
  def on_train_begin(self, logs={}):
        self.i = 0
        
  def on_epoch_end(self, epoch, logs={}):
    y_pred = self.model.predict(self.validation_data[0])
    print('prediction: {} at epoch: {}'.format(y_pred, epoch))