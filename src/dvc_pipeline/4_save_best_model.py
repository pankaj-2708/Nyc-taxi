import mlflow
import pandas as pd
import mlflow.sklearn
from pathlib import Path
import pickle
import joblib



def save_model(model, output_path):
    with open(output_path / "best_model", "wb") as f:
        pickle.dump(model, f)
        
def get_best_model():
    runs = mlflow.search_runs(experiment_names=["Default"])
    runs.sort_values(by="metrics.r2_score", ascending=False, inplace=True)
    best_run = runs.iloc[0]
    best_run_id=best_run['run_id']

    print("best_model_details")
    print(best_run)
    model_name = 'best_model'
    run_id=best_run_id
    model_uri = f'runs:/{run_id}/model'

    loaded_model=mlflow.pyfunc.load_model(model_uri=model_uri)
    return loaded_model

def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent

    output_path = home_dir / "model"
    output_path.mkdir(parents=True, exist_ok=True)
    model=get_best_model()
    save_model(model,output_path)
    
if __name__=="__main__":
    main()