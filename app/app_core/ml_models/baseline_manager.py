import joblib
import pandas as pd
from .base import ModelManager
from ..data_processing import text_extract


class BaselineManager(ModelManager):
    """
    Manager for baseline model.
    """

    FEATURES = [
        "wh_word_count",
        "sentence_count",
        "word_count",
        "example_count",
        "n_linebreaks",
        "title_word_count",
        "title_question_marks",
        "n_links",
        "n_tags",
        "n_lists"
    ]

    def __init__(self, model_path: str):
        self.__model_path = model_path
        self.set_extractor(text_extract.ManualFeatureExtract())
        self.__model = None
        self.load_model()

    def set_model_path(self, new_path: str):
        """
        Change the path to the model and load the new predictor.
        :param new_path:
        :return:
        """
        self.__model_path = new_path
        self.load_model()

    def load_model(self):
        """
        Loads saved model from disk.
        :return:
        """
        self.__model = joblib.load(self.__model_path)

    def make_prediction(self, title: str, body: str):
        """
        Makes prediction using the model.
        :param title: Title of question.
        :param body: Body of question.
        :return: Predicted label.
        """
        if self.__model is None:
            raise ValueError("Model not loaded!")

        data = pd.DataFrame({"title": [title], "body": [body], "tags": [""]})
        data = self.extractor.process_df(data)[self.FEATURES]
        return self.__model.predict(data)
