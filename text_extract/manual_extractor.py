import pandas as pd
from typing import List
from .base import Extractor
from pandarallel import pandarallel
from nltk import word_tokenize, sent_tokenize

pandarallel.initialize()


class ManualFeatureExtract(Extractor):
    """
    Class to extract some hand-engineered features from StackExchange text
    bodies given as rendered html.

    Usage:
    >>> out_df = ManualFeatureExtract.process_df(input_df)
    """

    BODY_COL: str = "body"  #: Name of column containing text body in the df
    TITLE_COL: str = "title"  #: Name of column containing title in the df
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
        "for instance",
        r"e\.g\."
    ]

    def __init__(self, data: pd.DataFrame):
        self.__data = data.copy()
        self.__words_vecs = None

    @property
    def df(self) -> pd.DataFrame:
        return self.__data

    @classmethod
    def process_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a dataframe to extract the text features.
        :param df: Input dataframe with a body and a title column.
        :return:
        """
        return cls(df).process_data()

    def tokenize_strings(self, column: str) -> pd.Series:
        """
        Tokenizes the strings in the given column (splits it into words).
        Steps:
        - Remove HTML tags
        - Remove non alphabetic characters
        - To lowercase
        :param column:
        :return:
        """
        tokenized = self.__data[column]\
            .str.replace(r"<[^<>]+>", " ")\
            .str.lower() \
            .str.replace(r"[^a-z]", " ") \
            .parallel_map(word_tokenize)
        return tokenized

    def word_vec(self) -> pd.Series:
        """
        Gives vector of tokenized question bodies.
        :return:
        """
        if self.__words_vecs is None:
            self.__words_vecs = self.tokenize_strings(self.BODY_COL)
        return self.__words_vecs

    def count_qu_marks(self, column: str) -> pd.Series:
        """
        Returns series of question mark counts in the given column.
        Consecutive question marks are counted as one.
        :param column:
        :return:
        """
        out = self.__data[column]\
            .str\
            .count(r"\?+")
        return out

    def body_question_marks(self):
        """
        Count question marks in body.
        :return:
        """
        self.__data["num_question_marks"] = self.count_qu_marks(self.BODY_COL)

    def title_question_marks(self):
        """
        Count question marks in title.
        :return:
        """
        self.__data["title_question_marks"] = \
            self.count_qu_marks(self.TITLE_COL)

    def count_words(self):
        """
        Adds word count column to dataframe.
        :return:
        """
        self.__data["word_count"] = self.word_vec()\
            .parallel_map(len)

    def count_sentences(self):
        """
        Adds sentence count column to dataframe.
        Steps:
        - Remove HTML tags
        - Tokenize
        - Count
        :return:
        """
        self.__data["sentence_count"] = self.__data[self.BODY_COL] \
            .str \
            .replace(r"<[^<>]+>", " ") \
            .parallel_map(sent_tokenize)\
            .parallel_map(len)

    def count_word_occurences(self, words: List[str], key: str):
        """
        Count the occurences of words from a list in the question body.
        :param words:
        :param key:
        :return:
        """
        expression = "|".join([w + r"[\s\.,:;]*" for w in words])
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

    def count_title_words(self):
        """
        Adds title word count column to the dataframe.
        :return:
        """
        self.__data["title_word_count"] = \
            self.tokenize_strings(self.TITLE_COL)\
            .parallel_map(len)

    def process_data(self) -> pd.DataFrame:
        """
        Adds the hand-engineered features and returns the processed dataframe.
        :return:
        """
        self.body_question_marks()
        self.count_whs()
        self.count_sentences()
        self.count_words()
        self.count_examples()
        self.count_line_breaks()

        self.count_title_words()
        self.title_question_marks()
        return self.df
