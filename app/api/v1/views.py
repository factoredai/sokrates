from app_core import handlers
from .serializers import Question
from fastapi.responses import HTMLResponse
from fastapi import APIRouter, Form, Request
from fastapi.templating import Jinja2Templates


router = APIRouter(prefix="/v1")
templates = Jinja2Templates(directory="templates")


@router.post('/json')
async def main(form: Question):
    """
    Endpoint to compute the model predictions: can be consulted directly.
    :param form: Title and body of the question.
    """
    title, body = form.title, form.body
    handler = handlers.V1Handler()
    return handler.run_model(title, body)


@router.post('/', response_class=HTMLResponse)
async def main_internal(
        request: Request,
        title: str = Form(...),
        body: str = Form(...)):
    """
    Respond to and HTTP request for the model sent by the frontend.
    :param request:
    :param title:
    :param body:
    :return:
    """
    handler = handlers.V1Handler()
    res = handler.run_model(title, body)
    return templates.TemplateResponse(
        "dynamic template.html",
        {'request': request, 'title': title, 'body': body, 'veredict': res}
    )
