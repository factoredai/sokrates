from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import boto3
import json
import os
import sys

sys.path.insert(1, './app_core')

from ml_models.lime_dl_manager import LIME3InputsMgr

# Import credentials from a JSON file with properties "key_id" (for the access
# key ID) and "access_key" (for the secret access key):
with open('.env/credentials.json', 'r') as fopen:
    credentials = json.load(fopen)

# Import file's S3 Bucket and path from a JSON file with properties "Bucket"
# and "path":
with open('.env/S3 path.json', 'r') as fopen:
    s3_path = json.load(fopen)

# Define S3 client:
s3 = boto3.client(
    's3', aws_access_key_id=credentials['key_id'],
    aws_secret_access_key=credentials['access_key']
)

# Check if ``./model`` has the necessary files and download them if necessary:
model_files = set(os.listdir('./model'))
for file in ['explainer.dill', 'meta.json', 'model.h5', 'tokenizer.json']:
    if file not in model_files:
        try:
            s3_obj = s3.get_object(
                Bucket=s3_path['Bucket'],
                Key='/'.join([s3_path['path'], file])
            )
        except boto3.resource('s3').meta.client.exceptions.NoSuchKey:
            raise Exception('Could not find "{}"'.format(file))
        with open(os.path.join('model', file), 'wb') as fopen:
            for chunk in s3_obj['Body'].iter_chunks():
                fopen.write(chunk)

manager = LIME3InputsMgr('./model')


class Form(BaseModel):
    title: str
    body: str


app = FastAPI()


@app.post('/')
async def main(form: Form):
    '''
    ``form`` is a Pydantic model containing the title and body sent in a post
    request
     '''
    from_dict = form.dict()
    title = from_dict['title']
    body = from_dict['body']

    prediction, explanation = manager.make_prediction(title, body)

    # return {'prediction': prediction, 'explanation': explanation}
    return {'prediction': float(prediction[0, 0]), **dict(explanation)}


    # try:

    #     file = s3.get_object(Bucket=s3_path['Bucket'], Key=os.path.join(s3_path['path'], filename + '.csv'))
    # except boto3.resource('s3').meta.client.exceptions.NoSuchKey:
    #     return JSONResponse(status_code=404, content={'message': 'File not found'})
    # except:
    #     return JSONResponse(status_code=500, content={'message': 'Unexpected error'})

    # return StreamingResponse(file['Body'].iter_chunks(), media_type='text/csv')

    if __name__ == '__main__':
        import uvicorn
        uvicorn.run("main:app", port=3002, reload=True)
