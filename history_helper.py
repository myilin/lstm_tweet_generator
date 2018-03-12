from matplotlib import pyplot
import csv

from filesystem_helper import getChartsPath, getModelPath

def plotHistory(model_name, timestamp):
    lines = []
    with open(getModelPath(model_name, timestamp) + 'history.log', 'r', newline='') as file_obj:
        lines = [line for line in csv.reader(file_obj, delimiter=',') if line][1:]
        lines = [(float(line[1]), float(line[2])) for line in lines]
        loss, val_loss = zip(*lines)
        pyplot.plot(list(loss))
        pyplot.plot(list(val_loss))
        pyplot.title(model_name)
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.savefig(getChartsPath() + timestamp + "-" + model_name)
        pyplot.savefig(getModelPath(model_name, timestamp) + "chart")
        pyplot.clf()

def getEpochsElapsed(model_name, timestamp):
    file_obj = open(getModelPath(model_name, timestamp) + 'history.log', 'r', newline='')
    history_lines = [line for line in csv.reader(file_obj, delimiter=',') if line]
    return max(len(history_lines)-1, 0)