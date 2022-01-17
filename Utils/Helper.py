import pathlib
import re
from os.path import join, splitext, sep
from typing import List


def natural_sort(in_list: List[str]) -> List[str]:
    """
    :param in_list: list of strings which have to be sort like human do
    :return: sorted list od strings
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(in_list, key=alphanum_key)


def convert_path(path: str) -> str:
    """
    :param path: path string
    :return: os friendly path
    """
    if isinstance(path, pathlib.PurePath):
        return path
    else:
        path_sep = "/" if "/" in path else "\\"
        new_path = join(*path.split(path_sep))
        if splitext(path)[1] is not None and not (
                path.endswith("/") or path.endswith("\\")):
            new_path = new_path + sep
        if path.startswith(path_sep):
            return sep + new_path
        else:
            return new_path


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def delete_leading_zero(num):
    if not num.startswith("0"):
        return num
    else:
        return delete_leading_zero(num[1:])
