import os
import sys

import numpy as np
from nltk.tokenize import TweetTokenizer

from tweets_helper import getTweets
from filesystem_helper import getDataPath

sys.path.insert(0, '../glove_twitter_tokenizer')
import preprocess_twitter

class WordSequenceProvider:
    def initialize(self, full_text):
        self.vec_len = 102
        print('Indexing word vectors.')
        self.embeddings_index = {}
        #self.embeddings_words = []
        with open('D:\\glove.twitter.27B\\glove.twitter.27B.100d.txt', encoding="utf-8") as f:
            max_count = 10000
            count = 0
            for line in f:
                values = line.split()
                word = values[0]
                #embeddings_words.append(word)
                coefs = np.asarray(values[1:]+[0.0,0.0], dtype='float32')
                self.embeddings_index[word] = coefs
                count += 1
                if(count > 10000):
                    break

        eot_vec = np.zeros(self.vec_len, dtype='float32')
        eot_vec[-2] = 1.0
        self.embeddings_index['<eot>'] = eot_vec

        self.unknown_vec = np.zeros(self.vec_len, dtype='float32')
        self.unknown_vec[-1] = 1.0

        print('Found %s word vectors.' %len(self.embeddings_index))

    def getSequences(self, text, maxlen):
        tokens = self.__tokenize(text)

        print('Vectorization...')
        token_vectors = np.zeros((len(tokens), self.vec_len), dtype='float32')

        unknown_words_count = 0
        for id, token in enumerate(tokens):
            if(token in self.embeddings_index):
                token_vectors[id] = np.array(self.embeddings_index[token])
            else:
                token_vectors[id] = np.array(self.unknown_vec)
                unknown_words_count += 1

        print('Found %s unknown tokens' %unknown_words_count)
        print('Miss rate: %f' %(unknown_words_count/len(tokens)))

        x = np.zeros((len(token_vectors), maxlen, self.vec_len), dtype='float32')
        y = np.zeros((len(token_vectors), self.vec_len), dtype='float32')

        step = 1

        for i in range(0, len(token_vectors) - maxlen, step):
            x[i] = token_vectors[i: i + maxlen]
            y[i] = token_vectors[i + maxlen]

        return x, y

    def __tokenize(self, text):
        print('Tokenizing text (%s characters)' %len(text))

        text = text.replace('\n---\n', " <eot> ")
        text = preprocess_twitter.tokenize(text)
        text = text.replace("<<", "<").replace(">>", ">")

        tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        tokens = tknzr.tokenize(text)

        print('Found %s tokens' %len(tokens))
        return tokens

    def generateText(self, model, seed_sentence, generated_text_size, maxlen, temperature=1.0):
        return "generateText not implemented"