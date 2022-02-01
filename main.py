import os
from pathlib import Path

import matplotlib.pyplot as plt
import typer as typer

from FeatureExtraction.MRMR import use_mrmr
from FeatureExtraction.PCA import build_dataset_with_pca
from LSTMModel.Model import training

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from LSTMModel.PrepareDataset import prepare_dataset
from PreProcessing.BuildDataset import build_dataset
from PreProcessing.FFT import fft_processing
from Utils.Constants import RAW_DATA_PATH, DEAP_ELECTRODES, SAVED_MODEL_GRAPH_PATH, SAVE_TRAINED_MODEL_PATH
from Utils.DataHandler import LoadData


def main(classify_type: str, fs: str, overwrite: bool):
    print(f"Run Training with {fs} as channel extraction method on {classify_type}")
    fs_pca = False
    fs_mrmr = False
    if classify_type == "Arousal":
        classify_name = "Arousal"
    else:
        classify_name = "Valence"

    if fs != "":
        classify_name = classify_name + "_" + fs

    if fs.lower() == "pca":
        fs_pca = True
    elif fs.lower() == "mrmr":
        fs_mrmr = True

    load_data = LoadData(RAW_DATA_PATH)
    for filename, data in load_data.yield_raw_data():
        fft_processing(subject=data,
                       filename=filename,
                       channels=range(len(DEAP_ELECTRODES)),
                       band=[4, 8, 12, 16, 25, 45],
                       window_size=256,
                       step_size=16,
                       sample_rate=128,
                       overwrite=overwrite,
                       fs=fs_pca or fs_mrmr)

    if fs.lower() == "pca":
        build_dataset_with_pca(participant_list=range(1, 33), components=20)
    elif fs.lower() == "mrmr":
        use_mrmr(participant_list=range(1, 33), components=20, classify_type=classify_type)
    else:
        build_dataset(participant_list=range(1, 33))

    x_train, y_train, x_test, y_test = prepare_dataset(classify_name, pca=fs_pca, mrmr=fs_mrmr)

    print("Training: ", x_train.shape, y_train.shape)
    print("Test: ", x_test.shape, y_test.shape)

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
    plt.savefig(Path(SAVED_MODEL_GRAPH_PATH, f"Graph_{classify_name}.pdf"))

    if not SAVE_TRAINED_MODEL_PATH.exists():
        SAVE_TRAINED_MODEL_PATH.mkdir(exist_ok=True)

    m.save(Path(SAVE_TRAINED_MODEL_PATH, f"fft_lstm_{classify_name}.h5"))

if __name__ == '__main__':
    typer.run(main)
