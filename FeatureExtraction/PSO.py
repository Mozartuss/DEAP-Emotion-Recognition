import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from FeatureExtraction.PSO_helper import jfs
from Utils.Constants import FINAL_DATASET_PATH_PSO, PREPROCESSED_DATA_PATH_FS, SAVE_PSO_CHANNELS_PATH

CLASSIFY_TYPE = "Valence"
N = 10  # number of particles
T = 10  # maximum number of iterations
COMP = 20


def use_pso(participant_list: [int] = range(1, 10), classify_type: str = "Arousal", n_particle: int = 128,
            n_iterations: int = 32, components: int = 20, n_cores: int = multiprocessing.cpu_count()):
    global CLASSIFY_TYPE, N, T, COMP
    COMP = components
    CLASSIFY_TYPE = classify_type
    N = n_particle
    T = n_iterations
    print(
        f"Run PSO channel selection method with {CLASSIFY_TYPE} and with {N} particles and with {T} max iterations on {n_cores} cores")

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    backup = []

    save_path_data_training = Path(FINAL_DATASET_PATH_PSO, "data_training.npy")
    save_path_label_training = Path(FINAL_DATASET_PATH_PSO, "label_training.npy")
    save_path_data_testing = Path(FINAL_DATASET_PATH_PSO, "data_testing.npy")
    save_path_label_testing = Path(FINAL_DATASET_PATH_PSO, "label_testing.npy")
    FINAL_DATASET_PATH_PSO.mkdir(exist_ok=True)
    SAVE_PSO_CHANNELS_PATH.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        for label, zx, p, ch, acc in executor.map(exec_pso, participant_list):
            print(f"Finish participant {p} with {len(ch)} channels")
            ch = list(ch)
            acc = list(acc)
            backup.append([p, ch, len(ch), acc])
            for i in range(0, zx.shape[0]):
                if i % 4 == 0:
                    x_test.append(zx[i].reshape(zx.shape[1] * zx.shape[2], ))
                    y_test.append(label[i])
                else:
                    x_train.append(zx[i].reshape(zx.shape[1] * zx.shape[2], ))
                    y_train.append(label[i])

    pd.DataFrame(backup, columns=["participant", "channels", "n_channels", "channel_acc"], index=None).to_csv(
        Path(SAVE_PSO_CHANNELS_PATH, f"pso_channels_{classify_type}.csv"), index=False)

    np.save(str(save_path_data_training), np.array(x_train), allow_pickle=True, fix_imports=True)
    np.save(str(save_path_label_training), np.array(y_train), allow_pickle=True, fix_imports=True)

    np.save(str(save_path_data_testing), np.array(x_test), allow_pickle=True, fix_imports=True)
    np.save(str(save_path_label_testing), np.array(y_test), allow_pickle=True, fix_imports=True)


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
    if CLASSIFY_TYPE.lower() == "arousal":
        y = np.repeat(label[:, 0], num_frequencies)
    else:
        y = np.repeat(label[:, 1], num_frequencies)
    # split data into train & validation (70 -- 30)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, stratify=y)
    fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}
    # parameter  # k-value in KNN
    w = 0.9
    c1 = 2
    c2 = 2
    opts = {'fold': fold, 'N': N, 'T': T, 'w': w, 'c1': c1, 'c2': c2}
    fmdl = jfs(x, opts, COMP)
    sf = fmdl['sf']

    x = pd.DataFrame(x)
    data_new = x[sf].to_numpy()
    z = []
    for i in data_new.transpose(1, 0):
        z.append(i.reshape(-1, num_frequencies))
    zx = np.array(z).transpose((1, 0, 2))
    return label, zx, participant, sf, fmdl['df']


if __name__ == '__main__':
    print(multiprocessing.cpu_count())
    use_pso(participant_list=range(1, 10), classify_type="Arousal", n_particle=2, n_iterations=1, components=20,
            n_cores=1)
