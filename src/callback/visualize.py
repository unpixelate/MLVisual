#%%
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import sys
from utils import helpers
from callback.sample import PredictionCallback

(train_data, train_targets), (test_data, test_target) = boston_housing.load_data()

print(f'Training data : {train_data.shape}')
print(f'Training label : {train_targets.shape}')
# print(f'Training sample : {train_targets}')
# print(f'Training target sample : {test_target}')

from IPython.display import clear_output
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        clear_output(wait=True)
        if self.i % 10 == 0:
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.legend()
            plt.show()

reduce_lr = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
PlotLosses2 = PlotLosses()
s = PredictionCallback()

@helpers.define_model_fit(callbacks=[s,PlotLosses2],epochs=200,validation_split=0.2)
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse')
    return model
        
#%%
if __name__ == "__main__":
    from inspect import getfullargspec
    model = build_model()
    model.fit(train_data,train_targets)
    print(getfullargspec(model.fit)) # kwonlydefaults={....}

#FullArgSpec(
# args=['self', 'x', 'y', 'batch_size']
# , varargs=None, varkw='kwargs'
# , defaults=(None, None, None)
# , kwonlyargs=['epochs', 'verbose', 'callbacks', 'validation_split', 'validation_data', 'shuffle', 'class_weight', 'sample_weight', 'initial_epoch', 'steps_per_epoch', 'validation_steps']
# , kwonlydefaults={'epochs': 200, 'verbose': 1, 'callbacks': [<keras.callbacks.EarlyStopping object at 0x000002164E10B5F8>, <__main__.PlotLosses object at 0x000002164E10B668>]
# , 'validation_split': 0.0, 'validation_data': None, 'shuffle': True, 'class_weight': None, 'sample_weight': None, 'initial_epoch': 0, 'steps_per_epoch': None, 'validation_steps': None}, annotations={})


# %%
