from collections import namedtuple

Feature = namedtuple("Feature", ["regex", "token"])

features = [
    Feature(r"http[s]?://t\.co\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "token_url_tweeter"),
    Feature(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "token_url_other"),
    Feature(r"^RT\s(?:.+\s)?@\w+\:", "token_retweet"),  # "RT @user: (...)"
    Feature(r"^@\w+", "token_reply"),                   # "@user (...)"
    Feature(r"\s@\w+", "token_mention"),                # "(...) @user (...)"
    Feature(r"\s#\w+", "token_hashtag"),
    Feature(r"[\u263a-\U0001f645]+|(\:\)+|\:\(+|<3|\:\/|\:-\/|\:\||\:p|\:P)", "token_emoji"),
    Feature(r"[0-9]+", "token_number"),
]