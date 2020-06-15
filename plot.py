import os
import matplotlib.pyplot as plt
import numpy as np
from utils import data_parser
from utils.features import features
from utils.twitter_types import UserID, Tweet
from typing import Dict, List

def generate_plot(ax, data: Dict[UserID, List[Tweet]], truths: Dict[UserID, float], token: str):
    avg_bots = []
    avg_hums = []

    for user_id, tweets in data.items():
        if truths[user_id] > 0.5:
            avg_bots.append(data_parser.get_average(tweets, token) * 100)
        else:
            avg_hums.append(data_parser.get_average(tweets, token) * 100)

    ax.hist([avg_bots, avg_hums], bins=25, label=["bots", "humans"])
    ax.legend(loc="upper right")
    ax.set_title(f"{token}")

def main():
    data = data_parser.get_data(os.getcwd())
    truths = data_parser.get_truths(os.getcwd())

    feature_count = len(features)
    fig, axs = plt.subplots(feature_count, figsize=(10, 10))

    for i in range(feature_count):
        generate_plot(axs[i], data, truths, features[i].token)

    fig.tight_layout()
    plt.savefig("token_histograms.png")

if __name__ == "__main__":
    main()