import os
import pandas as pd
import numpy as np
import threading
import requests
import re
import html
import spacy
from transformers import pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer


os.makedirs("data", exist_ok=True)
PARAMS = {
    "format": "jsonv2",
    "addressdetails": "1",
    "limit": "1",
    "accept-language": "en",
}
URL = "https://nominatim.openstreetmap.org/search"


def get_file_name(root, threadID):
    return os.path.join(root, f"location_{threadID}.csv")


def get_url_num(text):
    return len(re.findall(r"(https?://[^\s]+)", text))


def get_hash_num(text):
    return len(re.findall(r"#[a-zA-Z0-9]+", text.lower()))


def get_mentions_num(text):
    return len(re.findall(r"@[A-Za-z0-9_]+", text))


def get_words_num(text):
    return len(re.findall(r"[A-Za-z_]+", text))


def get_digits_num(text):
    return sum(c.isdigit() for c in text)


def get_characters_num(text):
    return sum(c.isalpha() for c in text)


def get_uppercase_num(text):
    return sum(c.isupper() for c in text)


def encode_html(text):
    if not pd.isna(text):
        text = html.unescape(text)
        return re.sub(r"%20", " ", text)
    return text


def remove_trash(text):
    # text = re.sub(r"#[a-zA-Z0-9]+", "", text)
    text = re.sub(r"#", "", text)

    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"[0-9]", "", text)
    text = re.sub(r"(https?://[^\s]+)", "", text)

    return text.lower()


def generate_attribs(data):
    data["url_count"] = data["text"].apply(get_url_num)
    data["mentions_count"] = data["text"].apply(get_mentions_num)
    data["hash_count"] = data["text"].apply(get_hash_num)
    data["words_count"] = data["text"].apply(get_words_num)
    data["digits_count"] = data["text"].apply(get_digits_num)
    data["characters_count"] = data["text"].apply(get_characters_num)
    data["uppercase_count"] = data["text"].apply(get_uppercase_num)

    return data


def clean_data(data):
    data["text"] = data["text"].apply(remove_trash)

    data["text"] = data["text"].apply(encode_html)
    data["location"] = data["location"].apply(encode_html)
    data["keyword"] = data["keyword"].apply(encode_html)

    return data


class LocationThread(threading.Thread):
    def __init__(self, threadID, l, r, data, notify, root):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.l = l
        self.r = r
        self.data = data
        self.notify = notify
        self.root = root

    def run(self):
        with requests.Session() as session:
            session.params.update(PARAMS)

            for i in range(self.l, self.r + 1):
                location = self.data.loc[i, "location"]

                if not pd.isna(location):
                    try:
                        response = session.get(
                            URL, params={"q": location.lower().strip()}
                        )
                    except Exception as e:
                        print(f"Thread {self.threadID}: {e}")
                    try:
                        info = response.json()[0]
                        code = info["address"]["country_code"]
                    except Exception as e:
                        code = "other"
                else:
                    code = "unknown"

                with open(get_file_name(self.root, self.threadID), "a") as f:
                    f.write(f"{i};{code}\n")

                if (i - self.l) % self.notify == 0:
                    print(
                        f"{self.threadID} just made {i - self.l}. ({self.r - i} to go)",
                    )


class LocationPreprocessor:
    def __init__(self, keep=40):
        self.keep = keep
        self.countries = None

    def download(self, data, root, sep_to=3, notify=100):
        batch_size = int(np.ceil(data.shape[0] / sep_to))
        threads = [None] * sep_to

        for file in os.listdir(root):
            if file.startswith("location_"):
                os.remove(os.path.join(root, file))

        l = 0
        for i in range(sep_to):
            r = min(data.shape[0] - 1, l + batch_size)
            threads[i] = LocationThread(i, l, r, data, notify, get_file_name)
            l = r + 1

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def preprocess(
        self, data, fit=False, hot_start=False, root="test_data_loc", **kwargs
    ):
        if not hot_start:
            self.download(data, root=root, **kwargs)
        locations = None

        for file in sorted(os.listdir(root)):
            if file.startswith("location_"):
                new_batch = pd.read_csv(
                    os.path.join(root, file), sep=";", encoding="utf-8", header=None
                ).to_numpy()

                if locations is None:
                    locations = new_batch
                else:
                    locations = np.r_[locations, new_batch]

        locations = locations[:, 1]

        # check if all downloaded
        assert len(locations) == data.shape[0], "Not all locations were downloaded"

        if fit:
            self.countries = pd.DataFrame(locations).value_counts()[: self.keep].index
        else:
            assert self.countries is not None, "You need to call with fit=True first"

        for i in range(len(locations)):
            if locations[i] not in self.countries:
                locations[i] = "other_country"

        return locations


class SentimentAdder:
    def __init__(self):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english"
        )

    def get_sentiment(self, row):
        label = self.sentiment_pipeline(row)[0]
        return 1 if label["label"] == "POSITIVE" else 0

    def add(self, data):
        data["sentiment"] = data["text"].apply(self.get_sentiment)
        return data


class DataPreprocessor:
    def __init__(self, **kwargs):
        self.loc_encoder = LocationPreprocessor(**kwargs)
        self.sentiment_adder = SentimentAdder(**kwargs)
        self.imputer = SimpleImputer(strategy="most_frequent")
        self.nlp = spacy.load("en_core_web_sm")

    def get_text(self, row):
        row = self.nlp(row)
        return " ".join([token.lemma_ for token in row if token.is_alpha])

    def preprocess(self, data, fit=False, hot_start=False, **kwargs):
        data = generate_attribs(data)
        data = clean_data(data)

        data["location"] = self.loc_encoder.preprocess(
            data, fit=fit, hot_start=hot_start, **kwargs
        )
        data = self.sentiment_adder.add(data)

        if fit:
            data["keyword"] = self.imputer.fit_transform(
                data["keyword"].to_numpy().reshape(-1, 1)
            )
        else:
            data["keyword"] = self.imputer.transform(
                data["keyword"].to_numpy().reshape(-1, 1)
            )

        data["text"] = data["text"].apply(self.get_text)

        return data


class DataTokenizer:
    def __init__(self):
        self.text_tv = TfidfVectorizer()
        self.keyword_tv = TfidfVectorizer()
        self.loc_tv = TfidfVectorizer()

    def fit(self, data):
        self.text_tv = self.text_tv.fit(data.loc[:, "text"])
        self.keyword_tv = self.keyword_tv.fit(data.loc[:, "keyword"])
        self.loc_tv = self.loc_tv.fit(data.loc[:, "location"])

    def transform(self, data):
        text_tokens = self.text_tv.transform(data.loc[:, "text"]).toarray()
        keyword_tokens = self.keyword_tv.transform(data.loc[:, "keyword"]).toarray()
        location_tokens = self.loc_tv.transform(data.loc[:, "location"]).toarray()

        data = data.drop(axis=1, columns=["text", "keyword", "location", "id"])
        return np.concatenate(
            [data.to_numpy(), text_tokens, keyword_tokens, location_tokens], axis=1
        )


class XPipeline:
    def __init__(self, step1, step2):
        self.step1 = step1
        self.step2 = step2

    def transform(self, data, root="test_data"):
        data = self.step1.preprocess(data, root=root)
        data = self.step2.transform(data)

        return data


class FinalXPipeline:
    def __init__(self, k, pure_pipeline, rfe):
        self.k = k
        self.pure_pipeline = pure_pipeline
        self.rfe = rfe

    def transform(self, data, root="test_data"):
        data = self.pure_pipeline.transform(data, root=root)
        data = self.rfe.transform(data)
        return data
