import numpy as np
import pandas as pd
import datetime as dt
import json

from searchtweets import collect_results, gen_request_parameters, load_credentials


# https://github.com/yumoxu/stocknet-dataset
# https://developer.twitter.com/en/docs/tutorials/how-to-analyze-the-sentiment-of-your-own-tweets
# https://github.com/twitterdev/search-tweets-python/tree/v2


def load_tweets():
    with open('tweets/raw/response-20-21-@Apple.json') as f:
        data = json.load(f)
    data = np.array(data['results'])
    parsed = []
    for tweet in data:
        if tweet['lang'] == 'en':
            parsed.append({
                'id': tweet['id'],
                'created_at': dt.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y'),
                'text': tweet['text']
            })
    for t in parsed:
        print(t)


class Tweets:

    def __init__(self, company, lang):
        self.company = company
        self.lang = lang
        self.search_tweets_args = load_credentials("config.yaml",
                                                   yaml_key="search_tweets_v2",
                                                   env_overwrite=False)

    def collect_results(self, start, end):
        query = gen_request_parameters(
            f"{self.company} lang:{self.lang}",
            granularity=None,
            results_per_call=10,
            start_time=start,
            end_time=end,
            tweet_fields="created_at,public_metrics")
        tweets = collect_results(
            query,
            max_tweets=10,
            result_stream_args=self.search_tweets_args
        )
        return tweets

    def get_tweets(self, start_time, end_time):
        split_start = start_time
        while split_start <= end_time:
            split_end = split_start + dt.timedelta(days=14)
            if split_end >= end_time:
                split_end = end_time

            time_deadline = dt.datetime.now() - dt.timedelta(hours=1)
            if split_end >= time_deadline:
                split_end = time_deadline

            print(split_start.date(), split_end.date())
            tweets = self.collect_results(str(split_start.date()), str(split_end.date()))
            for tweet in tweets:
                print(tweet)
            split_start = split_end
