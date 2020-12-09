from typing import List

MODEL_DIR: str = "/app/model"

REQ_FILES: List[str] = [
    "model.h5",
    "tokenizer.json",
    "meta.json",
    "explainer.dill"
]

ENVS = {
    "development": "development",
    "production": "production"
}
