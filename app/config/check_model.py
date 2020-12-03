import os
import boto3
from .constants import MODEL_DIR, REQ_FILES


def model_files_exist() -> bool:
    """
    Checks that all necessary model files exist.
    :return:
    """
    out = all([
        os.path.isfile(os.path.join(MODEL_DIR, p))
        for p in REQ_FILES
    ])
    return out


def check_model():
    """
    Checks if all the necessary model files are stored and downloads them from
    S3 if not.
    :return:
    """
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if model_files_exist():
        print("[INFO] Model ready!")
    else:
        print("[INfO] DOWNLOADING MODEL")
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        for fname in REQ_FILES:
            s3.download_file(
                os.getenv("BUCKET_NAME"),
                f"{os.getenv('MODEL_PATH')}/{fname}",
                os.path.join(MODEL_DIR, fname)
            )
        print("[INFO] Downloaded model!")

