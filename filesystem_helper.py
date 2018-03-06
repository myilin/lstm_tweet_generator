import os

def getPath(folder_name):
    output_root = "../deep_donald_data/"
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    path = output_root + folder_name + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def getChartsPath():
    return getPath("_charts")