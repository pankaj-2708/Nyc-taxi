from fastapi import FastAPI, Path, HTTPException, Query
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
    try:
        with open(path_, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        missing_pkg=MODEL_LIB_MAP[str(e).split("'")[1]]
        os.system(f"pip install {missing_pkg}")
        return load_pkl(path_)


def main():
    global model, transformer, ppl
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent
    print(home_dir)
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
    main()
    uvicorn.run(app,host='0.0.0.0',port=8000)
