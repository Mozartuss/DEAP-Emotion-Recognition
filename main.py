import os
from pathlib import Path
from pickle import dump

import matplotlib.pyplot as plt
import typer

from FeatureExtraction.CS import use_cs
from FeatureExtraction.GWO import use_gwo
from FeatureExtraction.MRMR import use_mrmr
from FeatureExtraction.PCA import build_dataset_with_pca
from FeatureExtraction.PSO import use_pso
from LSTMModel.Model import training

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from LSTMModel.PrepareDataset import prepare_dataset
from PreProcessing.BuildDataset import build_dataset
from PreProcessing.FFT import fft_processing
from Utils.Constants import RAW_DATA_PATH, DEAP_ELECTRODES, SAVED_MODEL_GRAPH_PATH, SAVE_TRAINED_MODEL_PATH, \
    SAVE_TRAINED_HISTORY_PATH
from Utils.DataHandler import LoadData


def main(classify_type: str = typer.Argument(..., help="The classification Type:\t Arousal or Valence"),
         fs: str = typer.Argument(..., help="The channel selection algorithm:\t PCA, MRMR, PSO, GWO, CS, NONE"),
         overwrite: bool = typer.Argument(...,
                                          help="If you want to build the dataset from scratch and overwrite all the previous files:\t True, False"),
         gpu: str = typer.Argument(...,
                                   help="Which GPU should be used for training, or with all GPUs?:\t 0,1,... or Multi")):
    print(f"Run Training with {fs} as channel extraction method on {classify_type}")
    if gpu.isnumeric():
        gpu_string = f"/device:GPU:{gpu}"
        typer.echo("Train on " + gpu_string)
        gpu_setting = tf.device(gpu_string)
    elif gpu.lower() == "multi":
        gpus = tf.config.list_logical_devices('GPU')
        if len(gpus) > 1:
            typer.echo("Train on multiple GPUS:")
            typer.echo(gpus)
            gpu_setting = tf.distribute.MirroredStrategy(gpus).scope()
        else:
            typer.echo("Train on single GPU: 0")
            gpu_setting = tf.device("/device:GPU:0")
    else:
        typer.echo("Train on CPU")
        gpu_setting = tf.device('/CPU:0')

    fs_pca = False
    fs_mrmr = False
    fs_pso = False
    fs_gwo = False
    fs_cs = False
    classify_name = classify_type.lower()

    if fs != "":
        classify_name = classify_name + "_" + fs

    if fs.lower() == "pca":
        fs_pca = True
    elif fs.lower() == "mrmr":
        fs_mrmr = True
    elif fs.lower() == "pso":
        fs_pso = True
    elif fs.lower() == "gwo":
        fs_gwo = True
    elif fs.lower() == "cs":
        fs_cs = True

    SAVED_MODEL_GRAPH_PATH.mkdir(exist_ok=True)
    SAVE_TRAINED_MODEL_PATH.mkdir(exist_ok=True)
    SAVE_TRAINED_HISTORY_PATH.mkdir(exist_ok=True)

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
                       fs=fs_pca or fs_mrmr or fs_pso or fs_gwo or fs_cs)

    if fs_pca:
        build_dataset_with_pca(participant_list=range(1, 33), components=20, classify_type=classify_type)
    elif fs_mrmr:
        use_mrmr(participant_list=range(1, 33), components=20, classify_type=classify_type)
    elif fs_pso:
        use_pso(participant_list=range(1, 33), classify_type=classify_type, n_particle=32, n_iterations=10,
                components=20)
    elif fs_gwo:
        use_gwo(participant_list=range(1, 33), classify_type=classify_type, n_particle=32, n_iterations=10,
                components=20)
    elif fs_cs:
        use_cs(participant_list=range(1, 33), classify_type=classify_type, n_particle=32, n_iterations=10,
               components=20)
    else:
        build_dataset(participant_list=range(1, 33))

    x_train, y_train, x_test, y_test = prepare_dataset(classify_name, pca=fs_pca, mrmr=fs_mrmr, pso=fs_pso, gwo=fs_gwo,
                                                       cs=fs_cs)

    print("Training: ", x_train.shape, y_train.shape)
    print("Test: ", x_test.shape, y_test.shape)

    try:
        with gpu_setting:
            h, m = training(y_train, y_test, x_train, x_test, 200)
            score = m.evaluate(x_test, y_test, verbose=1)
    except RuntimeError as e:
        print(e)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.plot(h.history['val_accuracy'])
    plt.plot(h.history['val_loss'])
    plt.title('test model')
    plt.legend(["accuracy", "loss"])
    plt.ylabel('loss/accuracy')
    plt.xlabel('epoch')
    plt.savefig(Path(SAVED_MODEL_GRAPH_PATH, f"Graph_{classify_name}.pdf"))

    m.save(Path(SAVE_TRAINED_MODEL_PATH, f"fft_lstm_{classify_name}.h5"))

    # saving the history of the model
    with open(Path(SAVE_TRAINED_HISTORY_PATH, f"fft_lstm_{classify_name}"), 'wb') as history:
        dump(h.history, history)


if __name__ == '__main__':
    typer.run(main)
