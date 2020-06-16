import pickle
from tempfile import gettempdir
from utils.preprocessor import nltk_cleanup, preprocess_tweets
from utils.twitter_types import UserID, Tweet
from xml.etree import ElementTree as ET
from os import listdir, path
from typing import Dict, List

def _get_training_path(cwd) -> str:
    return path.join(cwd, "pan19", "pan19-author-profiling-training-2019-02-18", "en")

def _get_tweets(training_path, file) -> List[str]:
    root = ET.parse(path.join(training_path, file)).getroot()
    return [d.text for d in root.findall(".//document")]

def get_average(tweets: List[Tweet], token: str) -> float:
    count = 0
    for tweet in tweets:
        count += tweet.count(token)

    return float(count) / len(tweets)

def get_data(cwd, use_cache: bool=True) -> Dict[UserID, List[Tweet]]:
    cache_path = path.join(gettempdir(), "preprocessed_tweets_cache")

    if path.exists(cache_path) and use_cache:
        return pickle.load(open(cache_path, "rb"))

    training_path = _get_training_path(cwd)
    data = {}

    with nltk_cleanup():
        for file in listdir(training_path):
            if ".xml" not in file:
                continue

            user_id = file.replace(".xml", "")
            data[user_id] = preprocess_tweets(_get_tweets(training_path, file))

    pickle.dump(data, open(cache_path, "wb"))
    return data

def get_truths(cwd, use_cache: bool=True) -> Dict[UserID, float]:
    cache_path = path.join(gettempdir(), "tweet_truths_cache")

    if path.exists(cache_path) and use_cache:
        return pickle.load(open(cache_path, "rb"))

    training_path = _get_training_path(cwd)
    truths = {}

    with open(path.join(training_path, "truth.txt"), "r") as f:
        for line in f:
            user_id, label, gender = line.split(":::")
            truths[user_id] = 1. if label == "bot" else 0.

    pickle.dump(truths, open(cache_path, "wb"))
    return truths