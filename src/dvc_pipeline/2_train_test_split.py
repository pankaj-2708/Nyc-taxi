import pathlib
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import yaml


def load_data(file_path):
    return pd.read_csv(file_path)


def split(df, test_size_, random_state_=42):
    train, test = train_test_split(df, test_size=test_size_, random_state=random_state_)
    return train, test



def save_data(train, test, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path / "train.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params = None
    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["dataset"]

    data_path = home_dir / "data" / "interim" / "transformed.csv"
    output_path = home_dir / "data" / "train_test_split"

    output_path.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path)
    train, test = None, None
    try:
        train, test = split(data, params["test_size"], params["random_state"])
    except KeyError:
        train, test = split(data, params["test_size"])

    save_data(train, test, output_path)


if __name__ == "__main__":
    main()
