from pathlib import Path

import numpy as np
import pyeeg as pe

from Utils.Constants import PREPROCESSED_DATA_PATH
from Utils.Helper import delete_leading_zero


def fft_processing(subject, filename, channels, band, window_size, step_size, sample_rate, overwrite):
    p_num = delete_leading_zero(filename.split(".")[0][1:])
    save_file_path = Path(PREPROCESSED_DATA_PATH, f"Participant_{p_num}.npy")
    if not save_file_path.exists() or overwrite:
        meta = []
        for i in range(0, 40):
            # loop over 0-39 trails
            data = subject["data"][i]
            # Arousal and Valence
            labels = subject["labels"][i][:2]
            start = 0

            while start + window_size < data.shape[1]:
                meta_array = []
                meta_data = []  # meta vector for analysis
                for j in channels:
                    # Slice raw data over 2 sec, at interval of 0.125 sec
                    x = data[j][start: start + window_size]
                    # FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
                    y = pe.bin_power(x, band, sample_rate)
                    meta_data = meta_data + list(y[0])

                meta_array.append(np.array(meta_data))
                label_bin = np.array(labels >= 5).astype(int)
                meta_array.append(label_bin)

                meta.append(np.array(meta_array, dtype=object))
                start = start + step_size

        meta = np.array(meta)
        if not PREPROCESSED_DATA_PATH.exists():
            PREPROCESSED_DATA_PATH.mkdir(exist_ok=True)

        np.save(save_file_path, meta, allow_pickle=True, fix_imports=True)


if __name__ == '__main__':
    from Utils.Constants import RAW_DATA_PATH, DEAP_ELECTRODES
    from Utils.DataHandler import LoadData

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
