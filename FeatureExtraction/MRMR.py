from pathlib import Path

import numpy as np
import pandas as pd
from mrmr import mrmr_classif

from Utils.Constants import FINAL_DATASET_PATH_MRMR, PREPROCESSED_DATA_PATH_FS, SAVE_MRMR_CHANNELS_PATH


def use_mrmr(participant_list=range(1, 33), components=20, classify_type: str = "Arousal"):
    print(f"Run MRMR channel selection method with {classify_type} to select {components} channels")

    num_channel = 32
    num_frequencies = 5

    save_path_data_training = Path(FINAL_DATASET_PATH_MRMR, "data_training.npy")
    save_path_label_training = Path(FINAL_DATASET_PATH_MRMR, "label_training.npy")
    save_path_data_testing = Path(FINAL_DATASET_PATH_MRMR, "data_testing.npy")
    save_path_label_testing = Path(FINAL_DATASET_PATH_MRMR, "label_testing.npy")

    FINAL_DATASET_PATH_MRMR.mkdir(exist_ok=True)
    SAVE_MRMR_CHANNELS_PATH.mkdir(exist_ok=True)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    backup = []
    for participant in participant_list:
        filename = f"Participant_{participant}.npy"
        with open(Path(PREPROCESSED_DATA_PATH_FS, filename), "rb") as file:
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

        x = pd.DataFrame(x)
        y = pd.Series(y)

        mrmr_x_idx = mrmr_classif(X=x, y=y, K=components)

        backup.append([participant, mrmr_x_idx, len(mrmr_x_idx)])

        data_new = x[mrmr_x_idx].to_numpy()

        z = []
        for i in data_new.transpose(1, 0):
            z.append(i.reshape(-1, num_frequencies))

        zx = np.array(z).transpose((1, 0, 2))

        for i in range(0, zx.shape[0]):
            if i % 4 == 0:
                x_test.append(zx[i].reshape(zx.shape[1] * zx.shape[2], ))
                y_test.append(label[i])
            else:
                x_train.append(zx[i].reshape(zx.shape[1] * zx.shape[2], ))
                y_train.append(label[i])

    pd.DataFrame(backup, columns=["participant", "channels", "n_channels"], index=None).to_csv(
        Path(SAVE_MRMR_CHANNELS_PATH, f"mrmr_channels_{classify_type}.csv"), index=False)

    np.save(save_path_data_training, np.array(x_train), allow_pickle=True, fix_imports=True)
    np.save(save_path_label_training, np.array(y_train), allow_pickle=True, fix_imports=True)

    np.save(save_path_data_testing, np.array(x_test), allow_pickle=True, fix_imports=True)
    np.save(save_path_label_testing, np.array(y_test), allow_pickle=True, fix_imports=True)

    print("Dataset has been transformed with mRMR")


if __name__ == '__main__':
    use_mrmr()
