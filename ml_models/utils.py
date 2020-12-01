import pandas as pd
from typing import List
from lime.lime_tabular import LimeTabularExplainer


def make_explainer(
        df: pd.DataFrame,
        features: List[str],
        **kwargs) -> LimeTabularExplainer:
    """
    Makes a new explainer from training data.
    :param df: Dataframe of training data to make the explainer.
    :param features:
    :param kwargs: Additional arguments for explainer construction
    :return:
    """
    explainer = LimeTabularExplainer(
        df[features],
        feature_names=features,
        **kwargs
    )
    return explainer
