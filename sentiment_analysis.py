import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.layers.preprocessing.text_vectorization import LOWER_AND_STRIP_PUNCTUATION

import pandas as pd


# http://help.sentiment140.com/for-students


class SentimentAnalysisModel:

    def __init__(self, batch_size, seed, max_features, sequence_length, embedding_dim):
        self.batch_size = batch_size
        self.seed = seed
        self.max_features = max_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.vectorize_layer = TextVectorization(
            standardize=LOWER_AND_STRIP_PUNCTUATION,
            max_tokens=max_features,
            output_mode='int',
            output_sequence_length=sequence_length)
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def vectorize_text(self, text, label):
        text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text), label

    def create_datasets(self):
        raw_train_ds = pd.read_csv(
            'text_dataset/training.1600000.processed.noemoticon.csv',
            encoding="ISO-8859-1",
            engine='python',
            names=["Score", "Id", "Date", "Query", "User", "Text"])
        train_dataset = raw_train_ds.drop(columns=["Id", "Date", "Query", "User"]).sample(frac=1)
        train_dataset.Score = train_dataset.Score.replace({2: 0.5, 4: 1})

        train_dataset = train_dataset[:int(len(train_dataset) * 0.04)]

        raw_train_ds = train_dataset[:int(len(train_dataset) * 0.8)]
        raw_val_ds = train_dataset[int(len(train_dataset) * 0.8):]

        raw_test_ds = pd.read_csv(
            'text_dataset/testdata.manual.2009.06.14.csv',
            names=["Score", "Id", "Date", "Query", "User", "Text"])

        test_dataset = raw_test_ds.drop(columns=["Id", "Date", "Query", "User"])
        test_dataset.Score = test_dataset.Score.replace({2: 0.5, 4: 1})
        raw_test_ds = test_dataset

        raw_train_ds = tf.data.Dataset.from_tensor_slices(
            (raw_train_ds.Text.values, raw_train_ds.Score.values)).shuffle(len(raw_train_ds)).batch(200)
        raw_val_ds = tf.data.Dataset.from_tensor_slices(
            (raw_val_ds.Text.values, raw_val_ds.Score.values)).shuffle(len(raw_val_ds)).batch(200)
        raw_test_ds = tf.data.Dataset.from_tensor_slices(
            (raw_test_ds.Text.values, raw_test_ds.Score.values)).batch(20)

        train_text = raw_train_ds.map(lambda x, y: x)
        self.vectorize_layer.adapt(train_text)

        self.train_ds = raw_train_ds.map(self.vectorize_text)
        self.val_ds = raw_val_ds.map(self.vectorize_text)
        self.test_ds = raw_test_ds.map(self.vectorize_text)

    def process(self):
        self.create_datasets()

        self.model = tf.keras.Sequential([
            layers.Embedding(self.max_features + 1, self.embedding_dim),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(1)])

        self.model.summary()

        self.model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                           optimizer='adam',
                           metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

        epochs = 15
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs)

        loss, accuracy = self.model.evaluate(self.test_ds)

        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

        self.model.save('saved_model/sa_model')

        self.model = tf.keras.Sequential([
            self.vectorize_layer,
            self.model,
            layers.Activation('sigmoid')
        ])

        self.model.compile(
            loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
        )

    def load_model(self):
        self.model = tf.keras.models.load_model('saved_model/sa_model')
        self.model.summary()

    def get_score(self, text):
        res = self.model.predict(text, verbose=1)
        print(res)
        return res
