"""The output of experiments is being stored in a separate folder,
specified in a data_folder_name variable.

We recommend regular backup of this folder to ensure that valuable results are not lost,
since they often take long time to produce.

However, avoid keeping it in the same repository with source code.

The structure of the output folder is as following:

../<data_folder_name>
|   +-- _charts
|   |   +-- <timestamp>-<model_name>.png
|   +-- <model_name>
|   |   +-- <timestamp>
|   |   |   +-- chart.png
|   |   |   +-- error.log
|   |   |   +-- history.log
|   |   |   +-- model.h5
|   |   |   +-- zepoch_<epoch_number>.txt
"""
import os

data_folder_name = "deepdonald_data"

def ensurePath(root, dir_names):
    """Creates hierarchy of nested folders starting from root, if they did not exist.
    Returns resulting path.
    """
    path = root
    for dir_name in dir_names:
        path += str(dir_name) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
    return path

def getModelPath(model_name, timestamp):
    """Returns path to a folder storing specified model/timestamp.
    Ensures its existence.
    """
    return ensurePath("../", [data_folder_name, model_name, timestamp])

def getChartsPath():
    """Returns path to charts folder.
    Ensures its existence.
    """
    return ensurePath("../", [data_folder_name, "_charts"])

def getLastTimestamp(model_name):
    """Returns the timestamp of when the model with a given name/config was last time created.
    Returns None if not found.
    """
    # Looking for a folder storing the given model.
    path = "../" + data_folder_name + "/" + model_name
    if not os.path.exists(path):
        return None

    # Assuming that all folder names in this directory are timestamps, 
    # return the latest one.
    return max(os.listdir(path))