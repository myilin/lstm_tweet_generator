import sys
import io
import traceback
import random
import datetime

import numpy as np

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from keras.models import load_model

from filesystem_helper import getModelPath
from history_helper import plotHistory, getEpochsElapsed
from tweets_helper import getTweets, shuffledTweets
from text_helper import getSequences
from generation_helper import generateText

def on_epoch_end(epoch, logs):
    plotHistory(model_name, timestamp)

    if(generate_on_epoch):
        text_file = open(getModelPath(model_name, timestamp) + "zepoch_" + str(epoch) + ".txt", 'w')
        
        for temperature in [0.2, 0.5, 1.0, 1.2]:
            text_file.write('\n----- temperature:' + str(temperature) + "\n")
            
            generated = generateText(
                model, seed_sentence, generated_text_size, maxlen, 
                chars, char_indices, indices_char, 
                temperature)
            
            text_file.write(generated)
        
        text_file.close()

    if(shuffle_on_epoch):
        shuffled_tweets = shuffledTweets(train_tweets)
        x, y = getSequences(shuffled_tweets, maxlen, chars, char_indices)
        np.copyto(train_x, x)
        np.copyto(train_y, y)


#random.seed(42)

# neural net training config
num_layers = 2 # (>=2)
num_neurons = 32
dropout = 0.5
batch_size = 10
learning_rate = 0.001
maxlen = 40

data_fraction = 1000
shuffle_on_epoch = False
total_epochs = 30

generate_on_epoch = True
generated_text_size = 200
seed_sentence = 'Lorem ipsum dolor sit amet orci aliquam.'

# generating unique model name
model_name = str(num_layers) + "x" + str(num_neurons)
model_name += "-" + str(dropout).replace('.', ',')
model_name += "-" + str(batch_size)
model_name += "-" + str(learning_rate).replace('.', ',')
model_name += "-" + str(data_fraction)
model_name += "-" + str(maxlen)

# use resume as a command line argument to continue interrupted training
resuming = False
if(len(sys.argv) > 1 and sys.argv[1] == "resume"):
    resuming = True
    print(resuming)

t = datetime.datetime.now()
timestamp = str(t.month) + "_" + str(t.day) + "-" + str(t.hour) + "h_" + str(t.minute) + "m"


try:
    train_tweets, test_tweets = getTweets(data_fraction)

    full_text = train_tweets + test_tweets
    chars = sorted(list(set(full_text + seed_sentence)))
    print('total chars:' + str(len(chars)))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    train_x, train_y = getSequences(train_tweets, maxlen, chars, char_indices)
    test_x, test_y = getSequences(test_tweets, maxlen, chars, char_indices)

    # build the model:
    print('Building model...')
    
    model = Sequential()
    
    model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, len(chars))))
    for i in range(num_layers - 2):
        model.add(LSTM(num_neurons, return_sequences=True, dropout=dropout))
    model.add(LSTM(num_neurons, dropout=dropout))
    
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    epochs_elapsed = 0

    model_path = getModelPath(model_name, timestamp) + 'model.h5'

    if(resuming):
        model = load_model(model_path)
        epochs_elapsed = getEpochsElapsed(model_name)
    
    csv_logger = CSVLogger(getModelPath(model_name, timestamp) + 'history.log', append = resuming)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    checkpointer = ModelCheckpoint(filepath = model_path, verbose=1, save_best_only=False)

    history = model.fit(train_x, train_y,
              batch_size = batch_size,
              epochs = total_epochs-epochs_elapsed,
              callbacks = [csv_logger, print_callback, checkpointer],
              validation_data = (test_x, test_y))
except:
    error_log_file = open(getModelPath(model_name, timestamp) + "error.log", "w")
    traceback.print_exc(file=error_log_file)
    error_log_file.close()
    
    raise