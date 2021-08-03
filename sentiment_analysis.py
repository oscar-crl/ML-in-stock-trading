import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.layers.preprocessing.text_vectorization import LOWER_AND_STRIP_PUNCTUATION

import numpy as np
import pandas as pd
import re
import string

# http://help.sentiment140.com/for-students
# https://developer.twitter.com/en/docs/tutorials/how-to-analyze-the-sentiment-of-your-own-tweets

batch_size = 32
seed = 42
max_features = 10000
sequence_length = 250
embedding_dim = 128


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


vectorize_layer = TextVectorization(
        standardize=LOWER_AND_STRIP_PUNCTUATION,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)


def model():
    raw_train_ds = pd.read_csv(
        'text_dataset/training.1600000.processed.noemoticon.csv',
        encoding="ISO-8859-1",
        engine='python',
        names=["Score", "Id", "Date", "Query", "User", "Text"])
    train_dataset = raw_train_ds.drop(columns=["Id", "Date", "Query", "User"])
    train_dataset.Score = train_dataset.Score.replace({2: 1, 4: 2})

    raw_train_ds = train_dataset[:int(len(train_dataset) * 0.8)]
    raw_val_ds = train_dataset[int(len(train_dataset) * 0.8):]

    raw_test_ds = pd.read_csv(
        'text_dataset/testdata.manual.2009.06.14.csv',
        names=["Score", "Id", "Date", "Query", "User", "Text"])

    test_dataset = raw_test_ds.drop(columns=["Id", "Date", "Query", "User"])
    test_dataset.Score = test_dataset.Score.replace({2: 1, 4: 2})
    raw_test_ds = test_dataset

    raw_train_ds = tf.data.Dataset.from_tensor_slices((raw_train_ds.Text.values, raw_train_ds.Score.values)).shuffle(len(raw_train_ds)).batch(4000)
    raw_val_ds = tf.data.Dataset.from_tensor_slices((raw_val_ds.Text.values, raw_val_ds.Score.values)).shuffle(len(raw_val_ds)).batch(4000)
    raw_test_ds = tf.data.Dataset.from_tensor_slices((raw_test_ds.Text.values, raw_test_ds.Score.values)).batch(20)

    for feat, targ in raw_train_ds.take(3):
        print('TRAIN Features: {}, Target: {}'.format(feat, targ))
    for feat, targ in raw_test_ds.take(3):
        print('TEST Features: {}, Target: {}'.format(feat, targ))

    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(3)])

    model.summary()

    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])

    epochs = 15
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy']
    )

    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)


    examples = [
        "The movie was great!",
        "The movie was okay.",
        "The movie was terrible..."
    ]
    res = export_model.predict(examples, verbose=1)
    print(res)
