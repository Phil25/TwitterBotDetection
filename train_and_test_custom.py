import sys
import os
from utils import data_parser
from utils.twitter_types import UserID, Tweet
from utils.features import features
from typing import Dict, List
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import layers

def to_feature_row(tweets: List[Tweet]) -> List[float]:
    row = []
    for feature in features:
        row.append(data_parser.get_average(tweets, feature.token))
    return row

def to_table(data: Dict[UserID, List[Tweet]], truths: Dict[UserID, float]) -> List[List[float]]:
    table = []
    for user_id, tweets in data.items():
        row = to_feature_row(tweets)
        row.append(truths[user_id])
        table.append(row)

    return table

def plot_histories(histories, x_len: int, y_len: int):
    scale = 6
    i = 0
    fig, axs = plt.subplots(x_len, y_len, figsize=(x_len * scale, y_len * scale))

    for x in range(x_len):
        for y in range(y_len):
            acc = [value * 100 for value in histories[i].history["accuracy"]]
            loss = [value * 100 for value in histories[i].history["loss"]]
            epochs = range(1, len(acc) + 1)

            axs[x, y].set_ylim((0, 100)) # limit y axis
            axs[x, y].set_title(f"Model {i+1}")

            axs[x, y].plot(epochs, acc, "g", label="Accuracy %")
            axs[x, y].plot(epochs, loss, "b", label="Loss %")

            # display resulting accuracy and loss
            axs[x, y].annotate(f"{acc[-1]:.2f}%", xy=(1, acc[-1]), xytext=(5, 0), xycoords=("axes fraction", "data"), textcoords="offset points")
            axs[x, y].annotate(f"{loss[-1]:.2f}%", xy=(1, loss[-1]), xytext=(5, 0), xycoords=("axes fraction", "data"), textcoords="offset points")

            axs[x, y].legend()

            i += 1

    fig.tight_layout()
    plt.savefig("learning_curves_custom.png")

def main(no_cache: bool):
    data = data_parser.get_data(os.getcwd(), use_cache=not no_cache)
    truths = data_parser.get_truths(os.getcwd(), use_cache=not no_cache)
    table = to_table(data, truths)

    print(f"Retrieved data ({len(data)} records)")

    x = [row[:-1] for row in table] # features
    y = [row[-1] for row in table]  # labels

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    models = []

    # Model 1
    models.append(Sequential([
        layers.Dense(12, input_dim=len(x[0]), activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]))

    # Model 2
    models.append(Sequential([
        layers.Embedding(5000, 32, input_length=len(x[0])),
        layers.Dropout(0.2),
        layers.LSTM(100),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]))

    # Model 3
    models.append(Sequential([
        layers.Flatten(input_shape=(len(x[0]),)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]))

    # Model 4
    models.append(Sequential([
        layers.Embedding(input_dim=188, output_dim=50, input_length=len(x[0])),
        layers.LSTM(100, activation="sigmoid", return_sequences=True),
        layers.Dropout(0.5),
        layers.LSTM(100, activation="sigmoid"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]))

    histories = []

    for i in range(len(models)):
        model = models[i]
        print("#" * 50)
        print(f"Training model #{i+1}")
        print("#" * 50)

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        histories.append(model.fit(x_train, y_train, epochs=50, batch_size=64))

        print(model.summary())
        model.evaluate(x_test, y_test)

    plot_histories(histories, 2, 2)

if __name__ == "__main__":
    main("--no-cache" in sys.argv)