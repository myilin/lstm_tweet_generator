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

from filesystem_helper import getPath
from history_helper import plotHistory, getEpochsElapsed
from tweets_helper import getTweets, shuffledTweets
from text_helper import getSequences

#random.seed(42)

t = datetime.datetime.now()
timestamp = str(t.month) + "_" + str(t.day) + "-" + str(t.hour) + "h_" + str(t.minute) + "m"
#timestamp = "2_22-10h_29m"

# neural net config
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

model_name = str(num_layers) + "x" + str(num_neurons)
model_name += "-" + str(dropout).replace('.', ',')
model_name += "-" + str(batch_size)
model_name += "-" + str(learning_rate).replace('.', ',')
model_name += "-" + str(data_fraction)
model_name += "-" + str(maxlen)
model_name += "-" + timestamp

train_tweets, test_tweets = getTweets(data_fraction)

full_text = train_tweets + test_tweets
chars = sorted(list(set(full_text)))

print(chars)
    
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

train_x, train_y = getSequences(train_tweets, maxlen, chars, char_indices)
test_x, test_y = getSequences(test_tweets, maxlen, chars, char_indices)

resuming = False
if(len(sys.argv) > 1 and sys.argv[1] == "resume"):
    resuming = True
    print(resuming)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    plotHistory(model_name)

    if(generate_on_epoch):
        text_file = open(getPath(model_name + "/generated") + "epoch_" + str(epoch) + ".txt", 'w')
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            text_file.write('\n----- diversity:' + str(diversity) + "\n")

            generated = ''
            sentence = 'Lorem ipsum dolor sit amet orci aliquam.'
            generated += sentence

            for i in range(generated_text_size):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char
            
            text_file.write(generated)
        text_file.close()

    if(shuffle_on_epoch):
        shuffled_tweets = shuffledTweets(train_tweets)
        x, y = getSequences(shuffled_tweets, maxlen, chars, char_indices)
        np.copyto(train_x, x)
        np.copyto(train_y, y)

print('total chars:' + str(len(chars)))
model = Sequential()

try:
    # build the model:
    print('Building model...')
    model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, len(chars))))
    for i in range(num_layers - 2):
        model.add(LSTM(num_neurons, return_sequences=True, dropout=dropout))
    model.add(LSTM(num_neurons, dropout=dropout))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    epochs_elapsed = 0

    model_path = getPath(model_name) + 'model.h5'

    if(resuming):
        model = load_model(model_path)
        epochs_elapsed = getEpochsElapsed(model_name)

    checkpointer = ModelCheckpoint(filepath = model_path, verbose=1, save_best_only=False)
    csv_logger = CSVLogger(getPath(model_name) + 'history.log', append = resuming)

    history = model.fit(train_x, train_y,
              batch_size = batch_size,
              epochs = total_epochs-epochs_elapsed,
              callbacks = [csv_logger, print_callback, checkpointer],
              validation_data = (test_x, test_y))
except:
    error_log_file = open(getPath(model_name) + "error.log", "w")
    traceback.print_exc(file=error_log_file)
    error_log_file.close()
    
    raise