import os

data_folder_name = "deepdonald_data"

def ensurePath(root, dir_names):
    path = root
    for dir_name in dir_names:
        path += str(dir_name) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
    return path

def getModelPath(model_name, timestamp):
    return ensurePath("../", [data_folder_name, model_name, timestamp])

def getChartsPath():
    return ensurePath("../", [data_folder_name, "_charts"])

def getLastTimestamp(model_name):
    path = "../" + data_folder_name + "/" + model_name
    if not os.path.exists(path):
        return None
    return max(os.listdir(path))