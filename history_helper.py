from matplotlib import pyplot
import csv

from filesystem_helper import getChartsPath, getModelPath

def plotHistory(model_name, timestamp):
    """Plots train/validation loss function history.

    History is being loaded from the history.log file in csv format from the folder corresponding
    to the model name and the timestamp when training started.
    
    Plots generated from the loss history are saved to char.png to the same folder,
    as well as to a common _chars/ folder, with a unique model-timestamp name
    (for easy search and chart comparison).
    """

    lines = []

    # Parameter newline='' is required fot csv reader. Refer to csv module documentation for more details.
    with open(getModelPath(model_name, timestamp) + 'history.log', 'r', newline='') as file_obj:
        
        # Reading all non-empty lines, except the first one (header).
        lines = [line for line in csv.reader(file_obj, delimiter=',') if line][1:]
        
        # Plotting second and third columns: Train loss, Validation loss.
        lines = [(float(line[1]), float(line[2])) for line in lines]
        loss, val_loss = zip(*lines)
        pyplot.plot(list(loss))
        pyplot.plot(list(val_loss))
        pyplot.title(model_name)
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')

        # Saving chart to two files (duplicates in different folders)
        pyplot.savefig(getChartsPath() + timestamp + "-" + model_name)
        pyplot.savefig(getModelPath(model_name, timestamp) + "chart")
        pyplot.clf()

def getEpochsElapsed(model_name, timestamp):
    """Loads the loss history file to find out how many epochs the specified model has been trained for."""

    file_obj = open(getModelPath(model_name, timestamp) + 'history.log', 'r', newline='')
    history_lines = [line for line in csv.reader(file_obj, delimiter=',') if line]
    
    # If no history lines or only a header found, return zero.
    return max(len(history_lines)-1, 0)