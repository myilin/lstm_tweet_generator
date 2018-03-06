import json
import random

tweets_delimiter = '\n---\n'

def getTweets(fraction=1):

    all_tweets_list = []

    for year in range(2014, 2019):
        file = open('../../data/tweets/condensed_' + str(year) + '.json', 'r')
        for tweet_object in json.load(file):
            all_tweets_list.append(str(tweet_object["text"].encode('ascii', 'ignore')))
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
