from abc import ABC, abstractmethod
from ..data_processing.text_extract.base import Extractor


class ModelManager(ABC):
    """
    Interface to handle ML models to make predictions.
    """

    __extractor: Extractor  #: Feature extractor (private)

    @abstractmethod
    def load_model(self):
        """
        Load a model from disk for making predictions.
        :return:
        """
        pass

    @abstractmethod
    def make_prediction(self, title: str, body: str):
        """
        Evaluate the model on the given inputs.
        :param title: Title of the question.
        :param body: Body of the question.
        :return:
        """
        pass

    def set_extractor(self, extractor: Extractor):
        """
        Setter for the extractor.
        :param extractor: A feature extractor.
        :return: None
        """
        self.__extractor = extractor

    @property
    def extractor(self) -> Extractor:
        """
        Feature extractor (getter).
        :return:
        """
        return self.__extractor
