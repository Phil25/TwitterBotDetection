import os
from utils import data_parser
from utils.twitter_types import UserID, Tweet
from utils.features import features
from typing import Dict, List
from sklearn import metrics
from sklearn.model_selection import train_test_split

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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(f"Dataset split (train: {len(y_train)}, test: {len(y_test)})")

    classifiers = [
        (RandomForestClassifier(), "Random Forest"),
        (AdaBoostClassifier(), "AdaBoost"),
        (SVC(kernel="linear", C=0.025), "Linear SVM"),
        (SVC(gamma=2, C=1), "RBF SVM"),
    ]

    print("Accuracies:")

    for clf, name in classifiers:
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"{name:>16} : {accuracy}")

if __name__ == "__main__":
    main()