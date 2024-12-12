import io
import json
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, Form, UploadFile
from img2feat import CNN
from onnxruntime import InferenceSession
from PIL import Image
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    cnn: str = "alexnet"
    threshold: float = 0.7


config = Config()

LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    models["cnn"] = CNN(config.cnn)
    with open("cifar10.onnx", "rb") as f:
        onx = f.read()
    models["session"] = InferenceSession(onx, providers=["CPUExecutionProvider"])
    with open("mean_scale.json", "r") as f:
        data = json.load(f)
    models |= data
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


class ImageRequest(BaseModel):
    file: UploadFile


@app.get("/status")
def read_root():
    return "ok"


@app.post("/prediction")
async def predict(req: Annotated[ImageRequest, Form()]):
    image = to_array(await req.file.read())

    # Feature Extraction (特徴量抽出)
    X = models["cnn"]([image])

    # Feature Standardization (特徴量の標準化)
    X = X - models["mean"]
    X = X / models["scale"]

    X = X.astype(np.float32)

    input_name = models["session"].get_inputs()[0].name  # is "X"
    # label_name = models["session"].get_outputs()[0].name  # is "output_label"
    label_name = models["session"].get_outputs()[1].name  # is "output_probability"

    proba = models["session"].run([label_name], {input_name: X})[0][0]
    print(proba)
    # >>> {0: 0.003050298895686865, 1: 0.010493496432900429, 2: 0.002061390085145831, 3: 0.17074337601661682, 4: 0.012792888097465038, 5: 0.6963527798652649, 6: 0.0022715579252690077, 7: 0.06515106558799744, 8: 0.005543916951864958, 9: 0.03153929114341736}

    proba_list = [(k, v) for k, v in proba.items()]
    proba_list = sorted(proba_list, key=lambda i: i[1], reverse=True)
    result = [{"label": LABELS[d[0]], "probability": d[1]} for d in proba_list]
    return {"result": result}


def to_array(file: bytes) -> np.array:
    img = Image.open(io.BytesIO(file))
    return np.array(img.convert("RGB"))
