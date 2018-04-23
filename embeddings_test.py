import sys
import io
import traceback
import random
import datetime

import numpy as np

from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras.regularizers import l1, l2, l1_l2

from filesystem_helper import getDataPath, getModelPath, getLastTimestamp
from history_helper import plotHistory, getEpochsElapsed
from tweets_helper import getTweets, shuffledTweets
from char_sequence_helper import CharSequenceProvider
from word_sequence_helper import WordSequenceProvider
from weights_helper import saveWeights

from nltk.tokenize import TweetTokenizer
from scipy.spatial import distance
from matplotlib import pyplot

sequence_provider = WordSequenceProvider()

train_tweets, test_tweets = getTweets(100)

full_text = train_tweets + test_tweets
sequence_provider.initialize(full_text)

vector = sequence_provider.vectorize(["president"])[0]

eucl_dist = []
cosine_dist = []

for word, vec in sequence_provider.embeddings_index.items():
    eucl_dist.append((word, distance.euclidean(vector, vec)))
    cosine_dist.append((word, distance.cosine(vector, vec)))

eucl_dist.sort(key=lambda v: v[1])
cosine_dist.sort(key=lambda v: v[1])

for i in range(20):
    print (str(eucl_dist[i][1]) + "  " + eucl_dist[i][0])
    print (str(cosine_dist[i][1]) + "  " + cosine_dist[i][0])
    print ("---\n")