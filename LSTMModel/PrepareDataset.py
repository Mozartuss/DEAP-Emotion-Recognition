from pathlib import Path
from sklearn.preprocessing import normalize, StandardScaler

from tensorflow.keras.utils import to_categorical

import numpy as np

from Utils.Constants import FINAL_DATASET_PATH


def prepare_dataset(label_type: str = "Arousal"):
    x_train = np.load(str(Path(FINAL_DATASET_PATH, "data_training.npy")))
    y_train = np.load(str(Path(FINAL_DATASET_PATH, "label_training.npy")))

    x_train = normalize(x_train)
    y_train_arousal = np.ravel(y_train[:, [0]])
    y_train_valence = np.ravel(y_train[:, [1]])
    y_train_arousal = to_categorical(y_train_arousal)
    y_train_valence = to_categorical(y_train_valence)
    x_train = np.array(x_train[:])

    x_test = np.load(str(Path(FINAL_DATASET_PATH, "data_testing.npy")))
    y_test = np.load(str(Path(FINAL_DATASET_PATH, "label_testing.npy")))

    x_test = normalize(x_test)
    y_test_arousal = np.ravel(y_test[:, [0]])
    y_test_valence = np.ravel(y_test[:, [1]])
    y_test_arousal = to_categorical(y_test_arousal)
    y_test_valence = to_categorical(y_test_valence)
    x_test = np.array(x_test[:])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    if label_type.lower() == "arousal":
        return x_train, y_train_arousal, x_test, y_test_arousal
    else:
        return x_train, y_train_valence, x_test, y_test_valence


if __name__ == '__main__':
    prepare_dataset()
