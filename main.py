from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import model

app = FastAPI()

origins = [
    "http://localhost:5173",
    "*"  # TODO. update this wildcard to website url.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Item(BaseModel):
    url: str


@app.post("/prediction")
def predict(item: Item):
    label, percentage, predictions_list = model.predict_image(item.url)
    return {
        "url": item.url,
        "label": label,
        "percentage": percentage,
        "predictions": predictions_list
    }
