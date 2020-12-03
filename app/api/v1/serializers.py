from pydantic import BaseModel


class Question(BaseModel):
    """
    Receive the title and body of a question for
    the model to evaluate.
    """
    title: str
    body: str
