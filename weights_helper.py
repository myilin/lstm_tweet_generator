import pandas as pd

from filesystem_helper import getDataPath, getModelPath

def saveWeights(model, model_name, timestamp, epoch):
    weights_file = open(getModelPath(model_name, timestamp) + "weights_" + str(epoch) + ".txt", 'w')
    for layer in model.layers:
        weights = layer.get_weights() # list of numpy arrays
        names = []
        data = []
        #print(len(weights))
        for id, w in enumerate(weights):
            #names.append(layer.__class__.__name__ + " " + str(id))
            #data.append(w)
            weights_file.write(layer.__class__.__name__ + " " + str(id) + "\n")
            df = pd.DataFrame(data=w)
            weights_file.write(str(df.describe()))
            weights_file.write("\n")
        
    weights_file.close()
        #df = pd.DataFrame.from_items(zip(names, data))
        #print(df.describe())