from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

early_stopper = EarlyStopping(patience=5)

def compile_model(network, input_shape):
    """

    :param network dict: dictionary with network parameters
    :param input_shape tuple: tuple with tradin data shape
    :return: compiled model
    """

    nb_layers = network.get('n_layers', 2)
    nb_neurons = network.get('n_neurons', 10)
    activation = network.get('activations', 'linear')
    optimizer = network.get('optimizers', 'adam')

    model = Sequential()

    model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
    model.add(Dropout(network.get('dropout', 1)))
    for i in range(nb_layers - 1):
        model.add(Dense(nb_neurons, activation=activation))
        model.add(Dropout(network.get('dropout', 1)))

    model.add(Dense(
        network.get('last_layer_neurons', 1),
        activation=network.get('last_layer_activations', 'linear'),
    ))

    model.compile(loss=network.get('losses', 'mse'), optimizer=optimizer)

    return model


def train_and_score(network, x_train, y_train, x_test, y_test):
    """

    :param network dict: dictionary with network parameters
    :param x_train array: numpy array with features for traning
    :param y_train array: numpy array with labels for traning
    :param x_test array: numpy array with labels for test
    :param y_test array: numpy array with labels for test
    :return float: score
    """

    model = compile_model(network, (x_train.shape[1],))

    model.fit(x_train, y_train,
              batch_size=network.get('batch_size', 128),
              epochs=10,  # using early stopping, so no real limit
              verbose=network.get('verbose', 0))
              #validation_data=(x_test, y_test),
              #callbacks=[early_stopper])

    #score = model.evaluate(x_test, y_test, verbose=0)
    
    y_pred = model.predict(np.array(x_test))
    y_pred = np.concatenate(y_pred)

    true = y_test
    pred =  y_pred

    print(' R2 = ', r2_score(true, pred))

    return r2_score(true, pred)
