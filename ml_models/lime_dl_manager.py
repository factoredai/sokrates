import os
import json
import pandas as pd
import tensorflow as tf
from typing import List, Tuple
from .base import ModelManager
from lime.lime_tabular import LimeTabularExplainer
from data_processing.text_extract import ManualFeatureExtract


class LIME3InputsMgr(ModelManager):
    """
    Manages a TF model that takes 3 inputs (title, body, hand-engineered
    features) and uses LIME to provide some basic suggestions.
    """

    FEATURES: List[str] = [
        "wh_word_count",
        "sentence_count",
        "word_count",
        "example_count",
        "n_linebreaks",
        "title_word_count",
        "title_question_marks",
        "num_question_marks",
        "n_links",
        "n_lists",
    ]

    def __init__(self, model_path: str):
        """
        :param model_path: Path to a directory that contains the model stored
            as h5, the tokenizer (as JSON) and the metadata JSON.
        """
        self.__model_path = model_path
        if not os.path.isdir(model_path):
            raise FileNotFoundError("Model directory not found!")
        self.__title_len = 0
        self.__body_len = 0
        self.__model = None
        self.__tokenizer = None
        self.load_model()
        self.set_extractor(ManualFeatureExtract())

    @property
    def model_path(self) -> str:
        return self.__model_path

    @property
    def title_len(self) -> int:
        """
        Length to which the question titles are to be padded.
        :return: Integer
        """
        return self.__title_len

    @property
    def body_len(self) -> int:
        """
        Length to which the question bodies are to be padded.
        :return: Integer
        """
        return self.__body_len

    def process_titles(self, titles: List[str]):
        """
        Tokenizes and pads a list of question titles.
        :param titles: List of titles as strings.
        :return: List of tokenized sequences.
        """
        seqs = tf.keras.preprocessing.sequence.pad_sequences(
            self.__tokenizer.texts_to_sequences(titles),
            maxlen=self.title_len,
            padding="post",
            truncating="post"
        )
        return seqs

    def process_bodies(self, bodies: List[str]):
        """
        Tokenizes and pads a list of question bodies.
        :param bodies: List of bodies as strings.
        :return: List of tokenized sequences.
        """
        seqs = tf.keras.preprocessing.sequence.pad_sequences(
            self.__tokenizer.texts_to_sequences(bodies),
            maxlen=self.body_len,
            padding="post",
            truncating="post"
        )
        return seqs

    def load_model(self):
        """
        Loads the model and tokenizer from disk.
        :return:
        """
        self.__model = tf.keras.models.load_model(
            os.path.join(self.model_path, "model.h5")
        )
        print("[INFO] Model loaded!")

        tok_pth = os.path.join(self.model_path, "tokenizer.json")
        with open(tok_pth, "r") as f:
            self.__tokenizer = tf.keras\
                .preprocessing\
                .text\
                .tokenizer_from_json(f.read())
        print("[INFO] Tokenizer loaded!")

        meta_pth = os.path.join(self.model_path, "meta.json")
        with open(meta_pth, "r") as f:
            meta = json.load(f)
            self.__title_len = meta["title_pad_length"]
            self.__body_len = meta["body_pad_length"]

    def df_to_inputs(self, df: pd.DataFrame) -> Tuple:
        """
        Turns a dataframe given by the extractor into the inputs expected by
        the model: a tuple (title_seqs, body_seqs, hand-engineered features).
        :param df: Dataframe with 'body', 'title' and 'tags' columns.
        :return:
        """
        title_input = self.process_titles(df["title"])
        body_input = self.process_bodies(df["body"])
        feat = df[self.FEATURES].copy()
        return title_input, body_input, feat

    def make_prediction(self, title: str, body: str):
        """
        Uses the model to make a prediction from the title and body of a
        question.
        :param title:
        :param body:
        :return:
        """
        data = pd.DataFrame({
            "title": [title],
            "body": [body],
            "tags": [""]
        })
        data = self.extractor.process_df(data)
        prediction = self.__model.predict(self.df_to_inputs(data))

        return prediction
