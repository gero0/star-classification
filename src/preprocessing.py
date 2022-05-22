import tensorflow as tf
from tensorflow.keras import layers

def dataframe_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop("StarType")
    df = {key: value[:, tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)

    return ds


def normalization_layer(name, dataset):
    normalizer = layers.Normalization(axis=None)
    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    normalizer.adapt(feature_ds)
    return normalizer


def cat_encoding_layer(name, dataset):
    index = layers.StringLookup()
    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
    return lambda feature: encoder(index(feature))

def encode_features(train_ds, normalize_numeric=True):
    all_inputs = []
    encoded_features = []

    num_param_headers = [
        "Temperature",
        "Luminosity",
        "Radius",
        "AbsoluteMagnitude",
    ]
    string_param_headers = ["StarColor", "SpectralClass"]

    for header in num_param_headers:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        all_inputs.append(numeric_col)  
        if normalize_numeric:
            norm_layer = normalization_layer(header, train_ds)
            enc_num_col = norm_layer(numeric_col)   
            encoded_features.append(enc_num_col)
        else:
            encoded_features.append(numeric_col)

    for header in string_param_headers:
        cat_col = tf.keras.Input(shape=(1,), name=header, dtype=tf.string)
        enc_layer = cat_encoding_layer(header, train_ds)
        enc_cat_col = enc_layer(cat_col)
        all_inputs.append(cat_col)
        encoded_features.append(enc_cat_col)

    return (all_inputs, encoded_features)

