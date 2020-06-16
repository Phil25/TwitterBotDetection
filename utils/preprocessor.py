import re
import nltk
import shutil
from os import path
from contextlib import contextmanager
from utils.features import Feature, features
from utils.twitter_types import Tweet
from typing import List

@contextmanager
def nltk_cleanup():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    yield

    for nltk_data in nltk.data.path:
        if path.exists(nltk_data):
            shutil.rmtree(nltk_data)

def _tokenize(tweets: List[str], feature: Feature) -> List[str]:
    return [re.sub(feature.regex, f" {feature.token} ", tweet).strip() for tweet in tweets]

def preprocess_tweets(tweets: List[str]) -> List[Tweet]:
    for feature in features:
        tweets = _tokenize(tweets, feature)

    return [nltk.word_tokenize(tweet) for tweet in tweets]