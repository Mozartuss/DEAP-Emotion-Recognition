import os
from pathlib import Path

import matplotlib.pyplot as plt

from LSTMModel.Model import training

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from LSTMModel.PrepareDataset import prepare_dataset
from PreProcessing.BuildDataset import build_dataset
from PreProcessing.FFT import fft_processing
from Utils.Constants import RAW_DATA_PATH, DEAP_ELECTRODES, SAVED_MODEL_GRAPH_PATH, SAVE_TRAINED_MODEL_PATH
from Utils.DataHandler import LoadData

if __name__ == '__main__':
    load_data = LoadData(RAW_DATA_PATH)
    for filename, data in load_data.yield_raw_data():
        fft_processing(subject=data,
                       filename=filename,
                       channels=range(len(DEAP_ELECTRODES)),
                       band=[4, 8, 12, 16, 25, 45],
                       window_size=256,
                       step_size=16,
                       sample_rate=128,
                       overwrite=True)

    build_dataset()

    y_train, y_test, x_train, x_test = prepare_dataset("Arousal")

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)

    h, m = training(y_train, y_test, x_train, x_test, 200)

    score = m.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if not SAVED_MODEL_GRAPH_PATH.exists():
        SAVED_MODEL_GRAPH_PATH.mkdir(exist_ok=True)

    plt.plot(h.history['val_accuracy'])
    plt.plot(h.history['val_loss'])
    plt.title('test model')
    plt.legend(["accuracy", "loss"])
    plt.ylabel('loss/accuracy')
    plt.xlabel('epoch')
    plt.savefig(Path(SAVED_MODEL_GRAPH_PATH, "Graph.pdf"))

    if not SAVE_TRAINED_MODEL_PATH.exists():
        SAVE_TRAINED_MODEL_PATH.mkdir(exist_ok=True)

    m.save(Path(SAVE_TRAINED_MODEL_PATH, "fft_lstm.h5"))
