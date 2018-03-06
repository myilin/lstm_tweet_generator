from matplotlib import pyplot
import csv

from filesystem_helper import getChartsPath, getPath

def plotHistory(model_name):
    lines = []
    with open(getPath(model_name) + 'history.log', 'r', newline='') as file_obj:
        lines = [line for line in csv.reader(file_obj, delimiter=',') if line][1:]
        lines = [(float(line[1]), float(line[2])) for line in lines]
        loss, val_loss = zip(*lines)
        pyplot.plot(list(loss))
        pyplot.plot(list(val_loss))
        pyplot.title(model_name)
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.savefig(getChartsPath() + model_name)
        pyplot.savefig(getPath(model_name) + "chart")
        pyplot.clf()

def getEpochsElapsed(model_name):
    try:
        file_obj = open(getPath(model_name) + 'history.log', 'r', newline='')
        return len([line for line in csv.reader(file_obj, delimiter=',') if line][1:])
    except:
        return 0