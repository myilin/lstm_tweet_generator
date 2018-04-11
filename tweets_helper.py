import json
import os
import random

tweets_delimiter = '\n---\n'

def getTweets(fraction=1):

    all_tweets_list = []

    for year in range(2009, 2030):
        path = '../trump_tweet_data_archive/condensed_' + str(year) + '.json'
        if os.path.exists(path):
            file = open(path, 'r')
            for tweet_object in json.load(file):
                all_tweets_list.append(str(tweet_object["text"].encode('utf-8', 'ignore'))
                    .replace("b'","")
                    .replace("--", " ")
                    .replace("-", " ")
                    .replace("x9c", ""))
            file.close()

    all_tweets_list = all_tweets_list[0:int(len(all_tweets_list)/fraction)]

    splitter_id = int(len(all_tweets_list)*0.8)
    train_tweets = all_tweets_list[0:splitter_id]
    test_tweets = all_tweets_list[splitter_id:]

    return tweets_delimiter.join(train_tweets), tweets_delimiter.join(test_tweets)

def shuffledTweets(tweets_text):

    tweets_list = tweets_text.split(tweets_delimiter)
    random.shuffle(tweets_list)
    return tweets_delimiter.join(tweets_list)
