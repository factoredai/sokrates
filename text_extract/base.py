import pandas as pd
from abc import ABC, abstractmethod


class Extractor(ABC):
    """
    Base class to represent feature extractors.
    """

    @abstractmethod
    def process_data(self) -> pd.DataFrame:
        """
        Extract features from a dataframe and return a df with the new
        columns.
        :return:
        """
        pass
