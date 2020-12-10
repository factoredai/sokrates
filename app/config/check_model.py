import os
import boto3
from .constants import MODEL_DIR, REQ_FILES, ENVS


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


def get_s3_client():
    """
    Builds an S3 client with the appropriate credentials setting. For
    development, the `ENV` variable must be set to `development` and the
    `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` variables must be set. For
    production / deployment, `ENV` must be set to `production` and the AWS
    credentials are not needed.
    :return:
    """
    if os.getenv("ENV", "") == ENVS["production"]:
        print("[INFO] Using production setup!")
        client = boto3.client("s3")
    else:
        print("[INFO] Using development setup!")
        client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
    return client


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
        s3 = get_s3_client()
        for fname in REQ_FILES:
            s3.download_file(
                os.getenv("BUCKET_NAME"),
                f"{os.getenv('MODEL_PATH')}/{fname}",
                os.path.join(MODEL_DIR, fname)
            )
        print("[INFO] Downloaded model!")

