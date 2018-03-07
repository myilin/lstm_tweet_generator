import os

def ensurePath(root, dir_names):
    path = root
    for dir_name in dir_names:
        path += str(dir_name) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
    return path

def getModelPath(model_name, timestamp):
    return ensurePath("../", ["deep_donald_data", model_name, timestamp])

def getChartsPath(model_name):
    return ensurePath("../", ["deep_donald_data", "_charts", model_name])