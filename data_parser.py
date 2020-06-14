from preprocessor import preprocess_tweets
from xml.etree import ElementTree as ET
from os import listdir, path
from typing import Dict, List
from twitter_types import UserID, Tweet

def _get_training_path(cwd) -> str:
    return path.join(cwd, "pan19", "pan19-author-profiling-training-2019-02-18", "en")

def _get_tweets(training_path, file) -> List[str]:
    root = ET.parse(path.join(training_path, file)).getroot()
    return [d.text for d in root.findall(".//document")]

def get_data(cwd) -> Dict[UserID, List[Tweet]]:
    training_path = _get_training_path(cwd)
    data = {}

    for file in listdir(training_path):
        if ".xml" not in file:
            continue

        user_id = file.replace(".xml", "")
        data[user_id] = preprocess_tweets(_get_tweets(training_path, file))

    return data

def get_truths(cwd) -> Dict[UserID, float]:
    training_path = _get_training_path(cwd)
    truths = {}

    with open(path.join(training_path, "truth.txt"), "r") as f:
        for line in f:
            user_id, label, gender = line.split(":::")
            truths[user_id] = 1. if label == "bot" else 0.

    return truths