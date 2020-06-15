import re
import nltk # TODO: nltk.download("punkt")
from utils.features import Feature, features
from utils.twitter_types import Tweet
from typing import List

def _tokenize(tweets: List[str], feature: Feature) -> List[str]:
    return [re.sub(feature.regex, f" {feature.token} ", tweet).strip() for tweet in tweets]

def preprocess_tweets(tweets: List[str]) -> List[Tweet]:
    for feature in features:
        tweets = _tokenize(tweets, feature)

    return [nltk.word_tokenize(tweet) for tweet in tweets]