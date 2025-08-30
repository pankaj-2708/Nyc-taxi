from fastapi import FastAPI, Path, HTTPException, Query
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
from utility import *
import os

app = FastAPI()

model = None
transformer = None
ppl = None


def load_pkl(path_):
    with open(path_, "rb") as f:
        return pickle.load(f)


def main():
    global model, transformer, ppl
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent.parent
    model_path = home_dir / "model" / "best_model.pkl"
    model = load_pkl(model_path)
    transformer_path = home_dir / "model" / "fn_y.pkl"
    transformer = load_pkl(transformer_path)
    ppl = load_pkl(home_dir / "model" / "ppl.pkl")




@app.get("/trip_duration")
def predict_duration(data: check_data):
    if model == None:
        raise HTTPException(status_code=500, detail="model not loaded")
    df = pd.DataFrame([data.dict()])
    df = process_data(df)
    X = ppl.transform(df)
    y = model.predict(X)
    return JSONResponse(
        content={
            "status_code": 200,
            "trip_duration_in_seconds": f"{transformer.inverse_transform(y)[0]}",
        }
    )

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8080)
    main()
