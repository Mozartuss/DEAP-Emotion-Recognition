import os
import pickle
import re
from os import listdir
from os.path import exists, isdir
from pathlib import Path

from tqdm import tqdm

from Utils.Helper import convert_path, natural_sort


class LoadData:

    def __init__(self, path):
        self.path = convert_path(path)
        self.__validate_path(path)

    @staticmethod
    def __validate_path(path: str) -> bool:
        try:
            if not isinstance(path, str) or not path:
                return False
            else:
                return exists(path)

        except TypeError:
            return False

    def yield_raw_data(self):
        filenames = None
        if isdir(self.path):
            filenames = next(os.walk(self.path))[2]
        else:
            os.mkdir(self.path)
            exit("Folder doesn't exist")
        filenames = natural_sort(filenames)

        if len(filenames) >= 1:
            pbar = tqdm(filenames, position=0)
            for file in pbar:
                if file.endswith(".dat"):
                    pbar.set_description("Reading %s" % file)
                    with open(Path(self.path, file), 'rb') as f:
                        yield file, pickle.load(f, encoding="latin1")

    def yield_csv_data(self, label=False):
        if isdir(self.path):
            root_path, dirs = next(os.walk(self.path))[:2]
            for dir in natural_sort(dirs):
                filenames = listdir(Path(root_path, dir))
                if len(filenames) >= 1:
                    if label:
                        filenames = [string for string in filenames if "label" in string]
                    else:
                        filenames = [string for string in filenames if "Trial" in string]
                    filenames = natural_sort(filenames)
                    pbar = tqdm(filenames, position=0)
                    for file in pbar:
                        pbar.set_description("Reading %s" % file)
                        with open(Path(root_path, dir, file)) as f:
                            _file = file.split(".")[0].split("_")
                            participant_number, trial_number = _file[::len(_file) - 1]
                            nums = re.findall(r"\d+", participant_number)
                            if nums:
                                participant_number = nums[0]
                            yield f, participant_number, trial_number
        else:
            print("Folder doesn't exist")
            exit(0)
