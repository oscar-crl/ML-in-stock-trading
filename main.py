import numpy
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import json

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import sentiment_analysis

import requests

# Load Data
company = 'AAPL'

USER_AGENT = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                   ' Chrome/91.0.4472.124 Safari/537.36')
    }
sesh = requests.Session()
sesh.headers.update(USER_AGENT)

# https://github.com/yumoxu/stocknet-dataset
# https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/#features
# https://developer.twitter.com/en/docs/tutorials/how-to-analyze-the-sentiment-of-your-own-tweets


def load_data(file, start, end):
    # fetched_data = pdr.DataReader(company, 'av-daily', start, end, api_key='6MD7RC1W9BX3339N')
    fetched_data = pdr.DataReader(company, 'yahoo', start, end, session=sesh)
    fetched_data.to_csv(f'data_{file}.csv')
    return fetched_data


def preprocess_tweets():
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


def main():
    print(f"Fetching data for {company}...\n")
    train_start = dt.datetime(2012, 1, 1)
    train_end = dt.datetime(2020, 1, 1)
    try:
        data = pd.read_csv('data_train.csv')
    except IOError:
        data = load_data('train', train_start, train_end)

    print(f"Daily closed stock price for {company}\n{data['Close']}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    try:
        test_data = pd.read_csv('data_test.csv')
    except IOError:
        test_data = load_data('test', test_start, test_end)

    actual_prices = test_data['Close'].values

    print(f"Actual Data for {company}\n{test_data['Close']}")
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot
    plt.plot(actual_prices, color='black', label=f'Actual {company} price')
    plt.plot(predicted_prices, color='green', label=f'Predicted {company} price')
    plt.title(f"{company} Share price")
    plt.xlabel("Time")
    plt.ylabel(f"{company} Share price")
    plt.legend()
    plt.show()

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction {prediction}")


if __name__ == '__main__':
    # main()
    # preprocess_tweets()
    sentiment_analysis.model()
