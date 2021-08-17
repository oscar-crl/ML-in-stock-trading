import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import requests

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers

import sentiment_analysis
from tweets import Tweets

USER_AGENT = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                   ' Chrome/91.0.4472.124 Safari/537.36')
}
sesh = requests.Session()
sesh.headers.update(USER_AGENT)


class StockModel:

    def __init__(self, company, prediction_days, plot, train_start, train_end, test_start, test_end):
        self.company = company
        self.prediction_days = prediction_days
        self.plot = plot
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

    def get_historical_prices(self, file, start, end):
        try:
            data = pd.read_csv(f'saved_datasets/data_{file}.csv')
            data = data.set_index("Date")
        except IOError:
            data = pdr.DataReader(self.company, 'yahoo', start, end, session=sesh)
            data.to_csv(f'saved_datasets/data_{file}.csv')
        return data

    def display_plot(self, actual_prices, predicted_prices):
        if self.plot:
            plt.plot(actual_prices, color='black', label=f'Actual {self.company} price')
            plt.plot(predicted_prices, color='green', label=f'Predicted {self.company} price')
            plt.title(f"{self.company} Share price")
            plt.xlabel("Time")
            plt.ylabel(f"{self.company} Share price")
            plt.legend()
            plt.show()

    def process(self):
        print(f"Fetching data for {self.company}...\n")
        data = self.get_historical_prices('train', self.train_start, self.train_end)
        data['Score'] = 0.2
        print(f"Daily closed stock price for {self.company}\n{data['Close']}")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        x_train = []
        y_train = []

        for x in range(self.prediction_days, len(scaled_data)):
            window_price = scaled_data[x - self.prediction_days:x, 0]
            window_score = data['Score'].values[x - self.prediction_days:x]
            x_train.append(list(zip(window_price, window_score)))
            y_train.append(scaled_data[x, 0])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        model = tf.keras.Sequential([
            layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 2)),
            layers.Dropout(0.2),
            layers.LSTM(units=50, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units=50),
            layers.Dropout(0.2),
            layers.Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=25, batch_size=32)

        test_data = self.get_historical_prices('test', self.test_start, self.test_end)
        test_data['Score'] = 0.8
        print(f"Actual Data for {self.company}\n{test_data['Close']}")

        total_dataset = pd.concat((data, test_data), axis=0)
        model_inputs = total_dataset["Close"][len(total_dataset) - len(test_data) - self.prediction_days:].values
        model_inputs = scaler.transform(model_inputs.reshape(-1, 1))

        x_test = []

        for x in range(self.prediction_days, len(model_inputs)):
            window_price = model_inputs[x - self.prediction_days:x, 0]
            window_score = total_dataset['Score'].values[x - self.prediction_days:x]
            x_test.append(list(zip(window_price, window_score)))

        x_test = np.array(x_test)

        actual_prices = test_data['Close'].values
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # Plot
        self.display_plot(actual_prices, predicted_prices)

        window_price = model_inputs[-self.prediction_days:, 0]
        window_score = total_dataset['Score'].values[-self.prediction_days:]

        real_data = [list(zip(window_price, window_score))]
        real_data = np.array(real_data)

        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        print(f"Prediction {prediction}")


def main():
    StockModel(
        company='AAPL',
        prediction_days=60,
        plot=True,
        train_start=dt.datetime(2019, 1, 1),
        train_end=dt.datetime(2020, 1, 1),
        test_start=dt.datetime(2020, 1, 1),
        test_end=dt.datetime.now()
    ).process()


def sa():
    sa = sentiment_analysis.SentimentAnalysisModel(
        batch_size=32,
        seed=42,
        max_features=10000,
        sequence_length=250,
        embedding_dim=128
    )
    sa.process()
    examples = [
        "This company is great!",
        "This company is okay.",
        "This company is terrible..."
    ]
    sa.get_score(examples)


def twt():
    Tweets(company='AAPL', lang='en').get_tweets(dt.datetime(2021, 8, 8), dt.datetime.now() - dt.timedelta(hours=1))


if __name__ == '__main__':
    main()
    # sa()
    # twt()
