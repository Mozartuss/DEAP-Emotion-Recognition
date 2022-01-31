from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from Utils.Constants import PREPROCESSED_DATA_PATH_PCA, FINAL_DATASET_PATH_PCA


def build_dataset_with_pca(participant_list=range(1, 33), components=20):
    save_path_data_training = Path(FINAL_DATASET_PATH_PCA, "data_training.npy")
    save_path_label_training = Path(FINAL_DATASET_PATH_PCA, "label_training.npy")
    save_path_data_testing = Path(FINAL_DATASET_PATH_PCA, "data_testing.npy")
    save_path_label_testing = Path(FINAL_DATASET_PATH_PCA, "label_testing.npy")

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for participant in participant_list:
        filename = f"Participant_{participant}.npy"
        with open(Path(PREPROCESSED_DATA_PATH_PCA, filename), "rb") as file:
            sub = np.load(file, allow_pickle=True)
            data = []
            label = []
            for i in range(0, sub.shape[0]):
                data.append(np.array(sub[i][0]))
                label.append(np.array(sub[i][1]))
            data = np.array(data)
            label = np.array(label)

        x = data.transpose((1, 0, 2)).reshape(32, -1).transpose((1, 0))

        pca = PCA(n_components=components)
        data_new = pca.fit_transform(x)

        z = []
        for i in data_new.transpose(1, 0):
            z.append(i.reshape(-1, 5))

        zx = np.array(z).transpose((1, 0, 2))

        for i in range(0, zx.shape[0]):
            if i % 4 == 0:
                x_test.append(zx[i].reshape(100, ))
                y_test.append(label[i])
            else:
                x_train.append(zx[i].reshape(100, ))
                y_train.append(label[i])

    if not FINAL_DATASET_PATH_PCA.exists():
        FINAL_DATASET_PATH_PCA.mkdir(exist_ok=True)

    np.save(save_path_data_training, np.array(x_train), allow_pickle=True, fix_imports=True)
    np.save(save_path_label_training, np.array(y_train), allow_pickle=True, fix_imports=True)
    print("training dataset:", np.array(x_train).shape, np.array(y_train).shape)

    np.save(save_path_data_testing, np.array(x_test), allow_pickle=True, fix_imports=True)
    np.save(save_path_label_testing, np.array(y_test), allow_pickle=True, fix_imports=True)
    print("testing dataset:", np.array(x_test).shape, np.array(y_test).shape)


if __name__ == '__main__':
    build_dataset_with_pca(participant_list=range(1, 33), components=20)
