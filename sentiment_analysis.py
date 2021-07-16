import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import os
import re
import string

batch_size = 32
seed = 42
max_features = 10000
sequence_length = 250
embedding_dim = 128


def set_dataset():
    url = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

    dataset = tf.keras.utils.get_file("stack_overflow_16k", url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')

    dataset_dir = os.path.join(os.path.dirname(dataset), 'stack_overflow')
    os.listdir(dataset_dir)


def create_subset(subset):
    return tf.keras.preprocessing.text_dataset_from_directory(
        'stack_overflow/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset=subset,
        seed=seed)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


if __name__ == '__main__':
    raw_train_ds = create_subset('training')
    raw_val_ds = create_subset('validation')
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'stack_overflow/test',
        batch_size=batch_size)

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(4)])

    model.summary()

    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])

    epochs = 15
    history = model.fit(
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
        loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)

    examples = [
        "public static void main(string[] args)",
        "console.log()",
        "for month in range(1, 13): print(i)"
    ]

    res = export_model.predict(examples, verbose=1)
    print(res)

