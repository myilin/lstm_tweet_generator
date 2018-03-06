#from logger import Logger
import sys
import numpy as np

def getSequences(text, maxlen, chars, char_indices):
    sys.stdout.write('corpus length:' + str(len(text)))

    # cut the text in semi-redundant sequences of maxlen characters
    
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    sys.stdout.write('nb sequences:' + str(len(sentences)))

    sys.stdout.write('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return x, y