import re
import nltk # TODO: nltk.download("punkt")
from features import Feature, features
from typing import List
from twitter_types import Tweet

def _tokenize(tweets: List[str], feature: Feature) -> List[str]:
    return [re.sub(feature.regex, f" {feature.token} ", tweet).strip() for tweet in tweets]

def preprocess_tweets(tweets: List[str]) -> List[Tweet]:
    for feature in features:
        tweets = _tokenize(tweets, feature)

    return [nltk.word_tokenize(tweet) for tweet in tweets]