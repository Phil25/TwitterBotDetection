import os
from utils import data_parser
from utils.twitter_types import UserID, Tweet
from utils.features import features
from utils.thirdparty import plot_learning_curve
from typing import Dict, List
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

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

def main():
    data = data_parser.get_data(os.getcwd())
    truths = data_parser.get_truths(os.getcwd())
    table = to_table(data, truths)

    print(f"Retrieved data ({len(data)} records)")

    x = [row[:-1] for row in table] # features
    y = [row[-1] for row in table]  # labels

    classifiers = [
        (RandomForestClassifier(), "Random Forest"),
        (AdaBoostClassifier(), "AdaBoost"),
        (SVC(kernel="linear", C=0.025), "Linear SVM"),
        (SVC(gamma=2, C=1), "RBF SVM"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    for i in range(len(classifiers)):
        clf, name = classifiers[i]
        plot_learning_curve(clf, name, x, y, axes=axes[:,i], ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    fig.tight_layout()
    plt.savefig("learning_curves.png")

if __name__ == "__main__":
    main()