from pydantic import BaseModel
from config.constants import MODEL_DIR
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app_core.ml_models.lime_parser import ExplanationParser
from app_core.ml_models.lime_dl_manager import LIME3InputsMgr

manager = LIME3InputsMgr(MODEL_DIR)


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
        request: Request,
        title: str = Form(...),
        body: str = Form(...)):
    """
    Respond to and HTTP request for the model sent by
    :param request:
    :param title:
    :param body:
    :return:
    """
    prediction, explanation = manager.make_prediction(title, body)
    res = ExplanationParser(prediction, explanation).get_string()
    return templates.TemplateResponse(
        "dynamic template.html",
        {'request': request, 'title': title, 'body': body, 'veredict': res}
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='127.0.0.1', port=3000, reload=True)
