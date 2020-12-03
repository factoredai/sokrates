from config.constants import MODEL_DIR
from .ml_models.lime_parser import ExplanationParser
from .ml_models.lime_dl_manager import LIME3InputsMgr


class V1Handler:
    """
    Class to handle requests for model evaluation using V1 API.
    """

    MANAGER = LIME3InputsMgr(MODEL_DIR)

    @classmethod
    def run_model(cls, title: str, body: str) -> str:
        """
        Runs the model and returns the suggestion.
        :param title:
        :param body:
        :return: The suggestion for improving the question.
        """
        prediction, explanation = cls.MANAGER.make_prediction(title, body)
        res = ExplanationParser(prediction, explanation).get_string()
        return res

