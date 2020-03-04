from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np

(train_data, train_targets), (test_data, test_target) = boston_housing.load_data()

print(f'Training data : {train_data.shape}')
print(f'Training label : {train_targets.shape}')
# print(f'Training sample : {train_targets}')
# print(f'Training target sample : {test_target}')

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])
    return model



def k_fold(k: int, train_data,train_targets,build_model,verbose=False):
    '''
    @ Input
        ** k: number of folds
        ** train_data , train_target: Training set
        ** build_model: function that returns a model (.fit and .evaluate be implemented)
    @ Output
        ** array of MAE scores for k fold
    '''
    num_val_samples = len(train_data) // k

    all_scores = []
    for i in range(k):
        print(f'Processing fold # {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
        
        partial_train_data = np.concatenate(
                                [train_data[:i * num_val_samples],
                                train_data[(i+1) * num_val_samples:]],
                                axis=0)
        partial_train_targets = np.concatenate(
                                [train_targets[:i * num_val_samples],
                                train_targets[(i+1)*num_val_samples:]],
                                axis=0)
        model = build_model()
        model.fit(partial_train_data,
                partial_train_targets,
                batch_size=1,
                verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        if verbose:
            print(f'fold#{i}: Validation MSE: {val_mse}. Validation MAE: {val_mae}')
        all_scores.append(val_mae)
    return all_scores

if __name__ == "__main__":
    k_fold(4,train_data,train_targets,build_model)
    pass