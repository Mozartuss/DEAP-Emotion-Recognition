import ast
from pathlib import Path

import pandas as pd

from Utils.Constants import SAVE_MRMR_CHANNELS_PATH, SAVE_PSO_CHANNELS_PATH

pso_ch = pd.read_csv(Path(SAVE_PSO_CHANNELS_PATH, "pso_channels.csv"))
mrmr_ch = pd.read_csv(Path(SAVE_MRMR_CHANNELS_PATH, "pso_channels.csv"))

for i in range(32):
    mrmr_channels = mrmr_ch.iloc[i]["channels"]
    mrmr_channels = ast.literal_eval(mrmr_channels)
    mrmr_channels.sort()
    pso_channels = pso_ch.iloc[i]["channels"]
    pso_channels = ast.literal_eval(pso_channels)
    pso_channels.sort()
    print(pso_channels)
    print(mrmr_channels)
    print("\n\n")
