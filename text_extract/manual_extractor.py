import pandas as pd
from typing import List
from .base import Extractor
from nltk import word_tokenize, sent_tokenize


class ManualFeatureExtract(Extractor):
    """
    Class to extract some hand-engineered features from StackExchange text
    bodies given as rendered html.
    """

    BODY_COL: str = "body"  #: Name of column containing text body in the df
    WH_WORDS: List[str] = [
        "what",
        "how",
        "where",
        "which",
        "when",
        "why",
        "who"
    ]

    EXAMPLE_ABBREV: List[str] = [
        "example",
        r"e\.g\."
    ]

    def __init__(self, data: pd.DataFrame):
        self.__data = data

    @property
    def df(self) -> pd.DataFrame:
        return self.__data

    def count_question_marks(self):
        """
        Adds num_question_marks column to dataframe. Consecutive question
        marks are counted as one.
        :return:
        """
        self.__data["num_question_marks"] = self.__data[self.BODY_COL]\
            .str\
            .count(r"\?+")

    def count_words(self):
        """
        Adds word count column to dataframe.
        :return:
        """
        # Regex version (might be faster)
        self.__data["word_count"] = self.__data[self.BODY_COL]\
            .str\
            .replace(r"<[^<>]+>", " ")\
            .str\
            .count(r"(?<=\s)[a-zA-Z]+(?=[^a-zA-Z]?)")

        # NLTK version
        # self.__data["word_count"] = self.__data[self.BODY_COL]\
        #     .map(word_tokenize)\
        #     .map(len)

    def count_sentences(self):
        """
        Adds sentence count column to dataframe.
        :return:
        """
        self.__data["sentence_count"] = self.__data[self.BODY_COL] \
            .str \
            .replace(r"<[^<>]+>", " ") \
            .map(sent_tokenize)\
            .map(len)

    def count_word_occurences(self, words: List[str], key: str):
        """
        Count the occurences of words from a list in the question body.
        :param words:
        :param key:
        :return:
        """
        expression = "|".join(words)
        colname = f"{key}_count"
        self.__data[colname] = self.__data[self.BODY_COL] \
            .str \
            .lower() \
            .str \
            .count(expression)

    def count_whs(self):
        """
        Add 'wh' word count column to the dataframe.
        :return:
        """
        self.count_word_occurences(self.WH_WORDS, "wh_word")

    def count_examples(self):
        """
        Count occurrences of the word 'example' or its abbreviations.
        :return:
        """
        self.count_word_occurences(self.EXAMPLE_ABBREV, "example")

    def count_line_breaks(self):
        """
        Count the number of line breaks in the question body.
        :return:
        """
        self.__data["n_linebreaks"] = self.__data[self.BODY_COL]\
            .str\
            .count(r"\n")

    def process_data(self) -> pd.DataFrame:
        """
        Adds the hand-engineered features and returns the processed dataframe.
        :return:
        """
        self.count_question_marks()
        self.count_whs()
        self.count_sentences()
        self.count_words()
        self.count_examples()
        self.count_line_breaks()
        return self.df
