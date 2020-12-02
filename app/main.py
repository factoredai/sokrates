from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import boto3
import json
import os
import sys

sys.path.insert(1, './app_core')

from ml_models.lime_dl_manager import LIME3InputsMgr
from ml_models.lime_parser import ExplanationParser

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


class Question(BaseModel):
    title: str
    body: str


app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/', response_class=HTMLResponse)
async def get_func():
    with open('./templates/bare template.html', 'rt') as fopen:
        return fopen.read()


@app.post('/json')
async def main(form: Question):
    '''
    ``form`` is a Pydantic model containing the title and body sent in a post
    request
     '''
    form_dict = form.dict()
    title = form_dict['title']
    body = form_dict['body']

    prediction, explanation = manager.make_prediction(title, body)
    res = ExplanationParser(prediction, explanation).get_string()
    return res


@app.post('/', response_class=HTMLResponse)
async def main_internal(
    request: Request, title: str = Form(...), body: str = Form(...)
):
    prediction, explanation = manager.make_prediction(title, body)
    res = ExplanationParser(prediction, explanation).get_string()
    return templates.TemplateResponse(
        "dynamic template.html",
        {'request': request, 'title': title, 'body': body, 'veredict': res}
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='127.0.0.1', port=3000, reload=True)
