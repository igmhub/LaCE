import os


def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder


def split_string(string):
    parts = string.split("_")
    return parts
