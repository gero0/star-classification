#TODO: delete later
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from preprocessing import prepare_data
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers

dataframe_original = pd.read_csv("6 class csv.csv")

features = dataframe_original.copy()
labels = features.pop("Star type")

(inputs, preprocessed_inputs) = prepare_data(features)

preprocessing_model = tf.keras.Model(inputs, preprocessed_inputs)

tf.keras.utils.plot_model(model=preprocessing_model , rankdir="LR", dpi=72, show_shapes=True)


#Creating model
classification_model_body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])

preprocessed_inputs = preprocessing_model(inputs)
result = classification_model_body(preprocessed_inputs)
model = tf.keras.Model(inputs, result)

model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam())

features_dict = {name: np.array(value) for name, value in features.items()}
model.fit(x=features_dict, y=labels, epochs=10)


