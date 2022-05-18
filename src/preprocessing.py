import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


def prepare_data(features):
    inputs = create_inputs(features)
    numeric_inputs, string_inputs = separate_inputs_by_type(inputs)

    normalized_numeric = normalize_numeric(features, numeric_inputs)
    processed_strings = string_lookup(features, string_inputs)

    all_features = [normalized_numeric]

    for string in processed_strings:
        all_features.append(string)

    preprocessed_inputs = layers.Concatenate()(all_features)

    return inputs, preprocessed_inputs


def create_inputs(features):
    inputs = {}
    for name, column in features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    return inputs


def separate_inputs_by_type(inputs):
    numeric_inputs = {
        name: input for name, input in inputs.items() if input.dtype == tf.float32
    }

    string_inputs = {
        name: input for name, input in inputs.items() if input.dtype == tf.string
    }

    return numeric_inputs, string_inputs


def normalize_numeric(features, inputs):
    concat = layers.Concatenate()(list(inputs.values()))
    normalize = layers.Normalization()
    normalize.adapt(np.array(features[inputs.keys()]))
    normalized_inputs = normalize(concat)
    return normalized_inputs


def string_lookup(features, inputs):
    processed_inputs = []
    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue

        lookup = layers.StringLookup(vocabulary=np.unique(features[name]))
        one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
        processed_inputs.append(one_hot(lookup(input)))

    return processed_inputs
