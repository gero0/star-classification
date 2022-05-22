import os
from unicodedata import name

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import preprocessing
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


def plot_history(history, name):
    plt.clf()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("./img/" + name + "_acc.png")
    # summarize history for loss
    plt.clf()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("./img/" + name + "_loss.png")


def create_model(inputs, encoded_features):
    all_features = tf.keras.layers.concatenate(encoded_features, name="Input")
    m = tf.keras.layers.Dense(6, activation="relu", name="Hidden")(all_features)
    # output layer - 6 categories of stars
    output = tf.keras.layers.Dense(6, activation="softmax", name="Output")(m)
    model = tf.keras.Model(inputs, output)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


random_state = 2137

dataframe = pd.read_csv("6 class csv.csv")
dataframe["StarColor"] = dataframe["StarColor"].str.upper()

train = dataframe.sample(frac=0.6, random_state=random_state)
remaining = dataframe.drop(train.index)

val = remaining.sample(frac=0.5, random_state=random_state)
test = remaining.drop(val.index)

print("TRAIN shape:", train.shape)
print("VAL shape:", val.shape)
print("TEST shape:", test.shape)

batch_sizes = [1, 2, 4, 8, 16, 32]

results = dict()

for batch_size in batch_sizes:
    print("Normalizing for batch size: ", batch_size)
    train_ds = preprocessing.dataframe_to_dataset(train, batch_size=batch_size)
    val_ds = preprocessing.dataframe_to_dataset(
        val, shuffle=False, batch_size=batch_size
    )
    test_ds = preprocessing.dataframe_to_dataset(
        test, shuffle=False, batch_size=batch_size
    )

    (inputs, encoded_features) = preprocessing.encode_features(train_ds, True)
    (nn_inputs, nn_encoded_features) = preprocessing.encode_features(train_ds, False)

    model = create_model(inputs, encoded_features)
    print("Training on  batch size: ", batch_size)
    tf.keras.utils.plot_model(model, rankdir="LR", show_layer_activations=True)
    history = model.fit(train_ds, epochs=500, validation_data=val_ds, verbose=0)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy_" + str(batch_size), accuracy)
    plot_history(history, "history_" + str(batch_size))
    model.save('./models/{}'.format(batch_size))

    model_nn = create_model(nn_inputs, nn_encoded_features)
    print("Training NN for batch size: ", batch_size)
    tf.keras.utils.plot_model(
        model_nn, rankdir="LR", to_file="model_nn.png", show_layer_activations=True
    )
    history_nn = model_nn.fit(train_ds, epochs=500, validation_data=val_ds, verbose=0)

    loss_nn, accuracy_nn = model_nn.evaluate(test_ds)
    print("Accuracy_nn_" + str(batch_size), accuracy_nn)
    plot_history(history_nn, "history_nn_" + str(batch_size))
    model_nn.save('./models/nn_{}'.format(batch_size))

    results[batch_size] = (accuracy, loss, accuracy_nn, loss_nn)

res_string = "Batch size,Accuracy,Loss,Accuracy(No Norm.),Loss(No Norm.)\n"
for key in results:
    result = results[key]
    line = "{},{},{},{},{}\n".format(key, result[0], result[1], result[2], result[3])
    res_string += line

with open("results.csv", 'w') as file:
    file.write(res_string)
