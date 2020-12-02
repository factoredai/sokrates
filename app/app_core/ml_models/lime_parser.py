import re
import numpy as np
from typing import List, Tuple


class ExplanationParser:

    THRESHOLD = 0.5
    COMPARE = re.compile(r"[<>]=?")  #: Regexp for comparison operators
    VAR_NAME = re.compile(r"[a-zA-Z_]+")  #: Regexp for variable names
    STR_TEMPLATE = (
        "This question could use some improvement! "
        "Try the following: {options}."
    )

    def __init__(
            self, prediction: np.ndarray,
            explanation: List[Tuple[str, float]]):

        self.__pred: float = prediction[0, 0]
        self.__expl_list: List[Tuple[str, float]] = explanation

    def get_largest_neg(self, number: int = 3) -> List[Tuple[str, float]]:
        """
        Get the variable items that have the most negative impact on the
        question.
        """
        sorted_params = sorted(
            [e for e in self.__expl_list if e[1] < 0],
            key=lambda p: p[1]
        )
        return sorted_params[:number]

    def parse_comparison(self, comp_string: str) -> str:
        """
        Parse a comparison string into a suggestion.
        """
        comparisons = self.COMPARE.findall(comp_string)
        if len(comparisons) == 0:
            return ""
        elif len(comparisons) > 1:
            verb = "change"
        elif "<" in comparisons[0]:
            verb = "increase"
        else:
            verb = "decrease"

        var_name = self.VAR_NAME.findall(comp_string)[0]
        return f"{verb} {var_name}"

    def get_string(self) -> str:
        """
        Gives the suggestion string.
        """
        if self.__pred > self.THRESHOLD:
            out = "This is a good question!"
        else:
            most_relevant = self.get_largest_neg()
            suggestions = [
                self.parse_comparison(k)
                for k, v in most_relevant
            ]
            out = self.STR_TEMPLATE.format(options=", ".join(suggestions))
        return out
