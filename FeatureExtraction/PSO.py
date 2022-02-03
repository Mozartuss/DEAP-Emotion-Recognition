import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from FeatureExtraction.PSO_helper import pso_main
from Utils.Constants import FINAL_DATASET_PATH_PSO, PREPROCESSED_DATA_PATH_FS, SAVE_PSO_CHANNELS_PATH

classify_type = "Valence"
N = 10  # number of particles
T = 10  # maximum number of iterations


def build_dataset_with_pso(participant_list=range(1, 10), ct: str = "Arousal", n_particle=32, n_iterations=100):
    global classify_type, N, T
    classify_type = ct
    N = n_particle
    T = n_iterations
    n_cores = multiprocessing.cpu_count()
    print(
        f"Run PSO channel selection method with {classify_type} and with {N} particles and with {T} max iterations on {n_cores} cores")

    backup = []

    SAVE_PSO_CHANNELS_PATH.mkdir(exist_ok=True)
    FINAL_DATASET_PATH_PSO.mkdir(exist_ok=True)

    for participant in participant_list:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        label, zx, p, ch, cost = exec_pso(participant)
        backup.append([p, ch, len(ch), cost])
        for i in range(0, zx.shape[0]):
            if i % 4 == 0:
                x_test.append(zx[i].reshape(zx.shape[1] * zx.shape[2], ))
                y_test.append(label[i])
            else:
                x_train.append(zx[i].reshape(zx.shape[1] * zx.shape[2], ))
                y_train.append(label[i])

        np.save(str(Path(FINAL_DATASET_PATH_PSO, f"Participant_{participant}", "data_training.npy")), np.array(x_train),
                allow_pickle=True, fix_imports=True)
        np.save(str(Path(FINAL_DATASET_PATH_PSO, f"Participant_{participant}", "label_training.npy")),
                np.array(y_train),
                allow_pickle=True, fix_imports=True)

        np.save(str(Path(FINAL_DATASET_PATH_PSO, f"Participant_{participant}", "data_testing.npy")), np.array(x_test),
                allow_pickle=True, fix_imports=True)
        np.save(str(Path(FINAL_DATASET_PATH_PSO, f"Participant_{participant}", "label_testing.npy")), np.array(y_test),
                allow_pickle=True, fix_imports=True)

    pd.DataFrame(backup, columns=["participant", "channels", "num_channel", "cost"], index=None).to_csv(
        Path(SAVE_PSO_CHANNELS_PATH, "pso_channels.csv"), index=False)


def exec_pso(participant):
    num_channel = 32
    num_frequencies = 5

    with open(Path(PREPROCESSED_DATA_PATH_FS, f"Participant_{participant}.npy"), "rb") as file:
        sub = np.load(file, allow_pickle=True)
        data = []
        label = []
        for i in range(0, sub.shape[0]):
            data.append(np.array(sub[i][0]))
            label.append(np.array(sub[i][1]))
        data = np.array(data)
        label = np.array(label)
    x = data.transpose((1, 0, 2)).reshape(num_channel, -1).transpose((1, 0))
    if classify_type.lower() == "arousal":
        y = np.repeat(label[:, 0], num_frequencies)
    else:
        y = np.repeat(label[:, 1], num_frequencies)
    # split data into train & validation (70 -- 30)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, stratify=y)
    fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}
    # parameter  # k-value in KNN

    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, "k": 30, "p": 2}
    dimensions = num_channel
    cost, pos = pso_main(x, y, options, dimensions, N, multiprocessing.cpu_count(), T)
    sel_index = np.asarray(range(0, dimensions))[pos == 1]
    x = pd.DataFrame(x)
    data_new = x[sel_index].to_numpy()
    z = []
    for i in data_new.transpose(1, 0):
        z.append(i.reshape(-1, num_frequencies))
    zx = np.array(z).transpose((1, 0, 2))
    return label, zx, participant, sel_index, cost


if __name__ == '__main__':
    print(multiprocessing.cpu_count())
    build_dataset_with_pso(participant_list=range(1, 33), ct="Arousal", n_particle=32, n_iterations=10)
