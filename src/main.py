import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import preprocessing
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def plot_history(history, name):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("./img/"+name+"_acc.png")
    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("./img/"+name+"_loss.png")

random_state = 2137

dataframe = pd.read_csv("6 class csv.csv")
dataframe["Star color"] = dataframe["Star color"].str.upper()

train = dataframe.sample(frac=0.5, random_state=random_state)
remaining = dataframe.drop(train.index)

val = remaining.sample(frac=0.3, random_state=random_state)
test = remaining.drop(val.index)

print("TRAIN shape:", train.shape)
print("VAL shape:", val.shape)
print("TEST shape:", test.shape)

batch_size = 8
train_ds = preprocessing.dataframe_to_dataset(train, batch_size=batch_size)
val_ds = preprocessing.dataframe_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = preprocessing.dataframe_to_dataset(test, shuffle=False, batch_size=batch_size)

(inputs, encoded_features) = preprocessing.encode_features(train_ds, True)

# model
all_features = tf.keras.layers.concatenate(encoded_features)
m = tf.keras.layers.Dense(6, activation="relu")(all_features)

# output layer - 6 categories of stars
output = tf.keras.layers.Dense(6, activation="softmax")(m)

model = tf.keras.Model(inputs, output)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

history = model.fit(train_ds, epochs=500, validation_data=val_ds)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

plot_history(history, "history")
