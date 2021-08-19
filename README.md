# ML-in-stock-trading
Application of Machine learning in stock trading. Prediction of stock prices using historical and social data. 
This is my final project of my Postgraduate year at the **University of Kent** for the program **MSc Advanced Computer Science Computational Intelligence**.

## Tech

##### Stack
- Python 3.8

##### Dependancies
- [Tensor Flow](https://www.tensorflow.org/)
- [Sklearn](https://scikit-learn.org)
- [Pandas Datareader](https://pandas-datareader.readthedocs.io/en/latest/)
- [Search-tweets](https://github.com/twitterdev/search-tweets-python/tree/v2)

## Usage

In `main.py`, the `main()` call the `StockModel` class.
The class pameters can be changed.
- `company` - The company ticker symbol
- `prediction_days` - How many days are used to predict the next day
- `plot` - Display or not the plot of actual and predicted data
- `train_start` `train_end` - Time period for the training data
- `test_start` `test_end` - Time period for the testing data
- `sentiment_analysis` - Text sentiment analysis class to get a score of the tweets

```sh
$ python main.py
```

##### Twitter API credentials
A Premium or Academic Track twitter dev account is required.
Credentials are required to be stored in a yaml file `config.yaml` and have a key `search_tweets_full_v2` as under:
```
search_tweets_full_v2:
  endpoint:  https://api.twitter.com/2/tweets/search/all
  consumer_key: <CONSUMER_KEY>
  consumer_secret: <CONSUMER_SECRET>
  bearer_token: <BEARER_TOKEN>
```

##### Datasets strcuture
This project need a folder structure to save and load saved dataset. You should create theses folders:
```
saved_datasets
  > stock_prices
  > text_sentiment
  > tweets
```

## Datasets
- **Stock prices** are gathered on [Yahoo](https://finance.yahoo.com/) Finance by [Pandas Datareader](https://pandas-datareader.readthedocs.io/en/latest/)
- **Text sentiment** datasets are from [Sentiment140](http://help.sentiment140.com/for-students)
- **Tweets** are gathered using the **Full-achrive endpoint** from the [Twitter API](https://developer.twitter.com/en/docs)
