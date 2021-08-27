import pandas as pd
import datetime as dt

from searchtweets import collect_results, gen_rule_payload, load_credentials


class Tweets:

    def __init__(self, company, lang, start_time, end_time, skip_days, sentiment_analysis, subset):
        self.company = company
        self.lang = lang
        self.search_tweets_args = load_credentials("config.yaml", yaml_key="search_tweets_full", env_overwrite=False)
        self.sentiment_analysis = sentiment_analysis
        self.subset = subset
        self.tweets_ds = self.get_tweets_dataset(start_time, end_time, skip_days)

    def collect_results(self, start, end):
        query = gen_rule_payload(
            f"{self.company} lang:{self.lang}",
            from_date=start,
            to_date=end,
            results_per_call=10,
        )
        tweets = collect_results(
            query,
            max_results=10,
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
                metrics = tweet.get('quoted_status')
                if not metrics:
                    metrics = tweet
                if metrics:
                    metrics_score = tweet["retweet_count"] + tweet["reply_count"] + tweet["favorite_count"] + tweet["quote_count"]
                    if metrics_score > selected_tweet['score']:
                        selected_tweet = {'score': metrics_score, 'text': tweet['text']}
            for days in pd.date_range(split_start, split_end):
                tweets_ds.loc[str(days.date())] = selected_tweet['text']
            split_start = split_end
        return tweets_ds

    def get_tweets_dataset(self, start_time, end_time, skip_days):
        try:
            data = pd.read_csv(f'saved_datasets/tweets/{self.subset}_raw_tweets.csv')
            data = data.set_index("Date")
        except IOError:
            data = self.get_tweets(start_time, end_time, skip_days)
            data.to_csv(f'saved_datasets/tweets/{self.subset}_raw_tweets.csv')
        return data

    def process(self):
        processed_tweets_ds = self.tweets_ds.apply(
            lambda x: pd.Series([x.Text, self.sentiment_analysis.get_score([x.Text])], index=['Text', 'Score']),
            axis=1)
        processed_tweets_ds.to_csv(f'saved_datasets/tweets/{self.subset}_processed_tweets.csv')
        return processed_tweets_ds
