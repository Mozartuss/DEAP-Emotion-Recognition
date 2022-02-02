from pathlib import Path

import numpy as np

from Utils.Constants import PREPROCESSED_DATA_PATH, FINAL_DATASET_PATH


def build_dataset(participant_list=range(1, 33)):
    save_path_data_training = Path(FINAL_DATASET_PATH, "data_training.npy")
    save_path_label_training = Path(FINAL_DATASET_PATH, "label_training.npy")
    save_path_data_testing = Path(FINAL_DATASET_PATH, "data_testing.npy")
    save_path_label_testing = Path(FINAL_DATASET_PATH, "label_testing.npy")

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for participant in participant_list:
        with open(Path(PREPROCESSED_DATA_PATH, f"Participant_{participant}.npy"), "rb") as file:
            sub = np.load(file, allow_pickle=True)
            for i in range(0, sub.shape[0]):
                if i % 4 == 0:
                    x_test.append(sub[i][0])
                    y_test.append(sub[i][1])
                else:
                    x_train.append(sub[i][0])
                    y_train.append(sub[i][1])

    if not FINAL_DATASET_PATH.exists():
        FINAL_DATASET_PATH.mkdir(exist_ok=True)

    np.save(save_path_data_training, np.array(x_train), allow_pickle=True, fix_imports=True)
    np.save(save_path_label_training, np.array(y_train), allow_pickle=True, fix_imports=True)

    np.save(save_path_data_testing, np.array(x_test), allow_pickle=True, fix_imports=True)
    np.save(save_path_label_testing, np.array(y_test), allow_pickle=True, fix_imports=True)

    print("Dataset has been transformed without any channel selection")


if __name__ == '__main__':
    build_dataset()
