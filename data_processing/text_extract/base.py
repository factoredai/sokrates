import pandas as pd
from abc import ABC, abstractmethod


class Extractor(ABC):
    """
    Interface to represent feature extractors.
    """

    @abstractmethod
    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the given dataframe to extract features.
        :param df:
        :return:
        """
        pass
