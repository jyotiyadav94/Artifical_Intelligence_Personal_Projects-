import json
import io
import os
from PIL import Image
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, Form
from fastapi import FastAPI, Request
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from copywritingAgent.router import agentMenu,agentSocialMedia,agentAdvertising,agentNewsletter
from fastapi import FastAPI, File, UploadFile, HTTPException


app = FastAPI()
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
print(templates_dir)
templates = Jinja2Templates(directory=templates_dir)

# Endpoint for Menu
@app.post("/menu")
async def process_menu(goal: str = Form(...)):
    try:
        response = agentMenu(goal)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for social media
@app.post("/socialMedia")
async def process_socialMedia(goal: str = Form(...), image: UploadFile = File(...)):
    try:
        request_object_content = await image.read()
        img = Image.open(io.BytesIO(request_object_content))
        response = agentSocialMedia(goal, img)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for advertising
@app.post("/advertising")
async def process_advertising(goal: str = Form(...), interest: str = Form(...)):
    try:
        response = agentAdvertising(goal, interest)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for newsletters
@app.post("/newsletter")
async def process_newsletter(goal: str = Form(...)):
    try:
        response = agentNewsletter(goal)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})