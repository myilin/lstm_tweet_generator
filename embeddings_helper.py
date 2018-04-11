import os
import sys

from nltk.tokenize import TweetTokenizer, WordPunctTokenizer

from tweets_helper import getTweets
from filesystem_helper import getDataPath

sys.path.insert(0, '../glove_twitter_tokenizer')
from preprocess_twitter import tokenize

GLOVE_DIR = 'D:\\glove.twitter.27B'

print('Indexing word vectors.')

#embeddings_index = {}
embeddings_words = []
with open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'), encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_words.append(word)
        #coefs = np.asarray(values[1:], dtype='float32')
        #embeddings_index[word] = coefs

print('Found %s word vectors.' %len(embeddings_words))#% len(embeddings_index))

embeddings_words = set(embeddings_words)

print('Found %s unique word vectors.' %len(embeddings_words))#% len(embeddings_index))

train_tweets, test_tweets = getTweets()
all_tweets = train_tweets + '\n---\n' + test_tweets
all_tweets = all_tweets.replace('\n---\n', " <eot> ")
all_tweets = tokenize(all_tweets)
all_tweets = all_tweets.replace("<<", "<").replace(">>", ">")

tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
tokens = list(set(tknzr.tokenize(all_tweets)))

tokens2 = []

for t in tokens:
    if('.' in t):
        tokens2.append(".")
        tokens2.extend(t.split("."))
    else:
        tokens2.append(t)

tokens = list(set(tokens2))

print('Found %s unique words in corpus.' %len(tokens))

unknown_tokens = [t + "\n" for t in tokens if t not in embeddings_words]

known_tokens = [t + "\n" for t in tokens if t in embeddings_words]

print('Found %s known words' %len(known_tokens))
print('Found %s new words (not seen in embeddings)' %len(unknown_tokens))

known_words_file = open(getDataPath() + "known_words.txt", "w")
known_words_file.writelines(sorted(known_tokens))
known_words_file.close()

new_words_file = open(getDataPath() + "new_words.txt", "w")
new_words_file.writelines(sorted(unknown_tokens))
new_words_file.close()