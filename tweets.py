import pandas as pd
import datetime as dt

from searchtweets import collect_results, gen_request_parameters, load_credentials


class Tweets:

    def __init__(self, company, lang, start_time, end_time, skip_days, sentiment_analysis):
        self.company = company
        self.lang = lang
        self.search_tweets_args = load_credentials("config.yaml", yaml_key="search_tweets_full_v2", env_overwrite=False)
        self.tweets_ds = self.get_tweets_dataset(start_time, end_time, skip_days)
        self.sentiment_analysis = sentiment_analysis

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

    def get_tweets(self, start_time, end_time, skip_days):
        tweets_ds = pd.DataFrame(columns=["Text"], index=pd.date_range(start_time, end_time))
        tweets_ds.index.name = "Date"
        split_start = start_time
        end_time = end_time - dt.timedelta(hours=3)
        while split_start < end_time:
            split_end = split_start + dt.timedelta(days=skip_days)
            if split_end >= end_time:
                split_end = end_time
            time_deadline = dt.datetime.now() - dt.timedelta(hours=3)
            if split_end >= time_deadline:
                split_end = time_deadline
                end_time = time_deadline
            tweets = self.collect_results(split_start.strftime("%Y-%m-%d %H:%M"), split_end.strftime("%Y-%m-%d %H:%M"))
            selected_tweet = {'score': 0, 'text': ""}
            for tweet in tweets:
                for t in tweet['data']:
                    metrics = t['public_metrics']
                    metrics_score = metrics["retweet_count"] + metrics["reply_count"] + metrics["like_count"] + metrics["quote_count"]
                    if metrics_score > selected_tweet['score']:
                        selected_tweet = {'score': metrics_score, 'text': t['text']}
            for days in pd.date_range(split_start, split_end):
                tweets_ds.loc[str(days.date())] = selected_tweet['text']
            split_start = split_end
        return tweets_ds

    def get_tweets_dataset(self, start_time, end_time, skip_days):
        try:
            data = pd.read_csv('saved_datasets/raw_tweets_ds.csv')
            data = data.set_index("Date")
        except IOError:
            data = self.get_tweets(start_time, end_time, skip_days)
            data.to_csv('saved_datasets/raw_tweets_ds.csv')
        return data

    def process(self):
        processed_tweets_ds = self.tweets_ds.apply(
            lambda x: pd.Series([x.Text, self.sentiment_analysis.get_score([x.Text])], index=['Text', 'Score']),
            axis=1)
        processed_tweets_ds.to_csv('saved_datasets/processed_tweets_ds.csv')
        return processed_tweets_ds
