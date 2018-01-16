import os
import warnings
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")            # Hide messy Numpy warnings

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def load_data(filename, normalise):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')
    data = [float(x) for x in data[:-1]]
    data = np.array(data)

    if normalise == 4:
        return data[3000:]
    else:
        result = []
        for step in range(len(data) - 3099):
            temp = data[step:step+3100]
            temp = np.mean(temp.reshape(-1, 100), axis=1)
            result.append(temp)

        result = np.array(result).astype(float)

        if normalise == 1:
            result = normalise_windows(result)
        elif normalise == 2:
            result = result / 10000
        elif normalise == 3:
            pass
        else:
            print("invalid normalize value")
            exit()

        result = np.array(result).astype(float)

        x_test = result[:, :-1]
        y_test = result[:, -1]

        return [x_test, y_test]

def build_model():
    model = Sequential()

    model.add(LSTM(
        input_dim=4,
        output_dim=75,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        150,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=4))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop")
    return model