import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer


def load_data(input_path):
    return pd.read_csv(input_path)


def save_data(df, output_path):
    df.to_csv(output_path / "transformed.csv", index=False)


def transform(df, normalise, output_path):
    clm_trans_cat = ColumnTransformer([("onehot", OneHotEncoder(), [-1])], remainder="passthrough")

    pf = PowerTransformer()
    fn_y = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

    clm_trans_num = None

    y = df["trip_duration"]
    fn_y.fit(y)
    with open(output_path / "fn_y.pkl", "wb") as f:
        pickle.dump(fn_y, f)

    df = df.drop(columns=["trip_duration"], axis=1)

    y = fn_y.transform(y)

    if normalise:
        clm_trans_num = ColumnTransformer(
            [
                ("std", MinMaxScaler(), [2, 3, 4, 5, 14, 15, 17, 18]),
                ("pf", pf, [7, 8, 9, 10, 11, 12, 13, 16]),
            ],
            remainder="passthrough",
        )
    else:
        clm_trans_num = ColumnTransformer(
            [
                ("std", MinMaxScaler(), [2, 3, 4, 5, 14, 15, 17, 18]),
                ("pf", pf, [7, 8, 9, 10, 11, 12, 13, 16]),
            ],
            remainder="passthrough",
        )

    ppl = Pipeline(steps=[("numerical", clm_trans_num), ("cat", clm_trans_cat)])

    ppl.fit(df, y)
    df = ppl.transform(df)

    with open(output_path / "ppl.pkl", "wb") as f:
        pickle.dump(ppl, f)

    new_df = pd.DataFrame(df)
    new_df["trip_duration"] = y
    return new_df


def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    input_path = home_dir / "data" / "processed" / "data.csv"

    output_path = home_dir / "data" / "interim"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["transform"]

    df = load_data(input_path)
    df = transform(df, params["normalise"], output_path)
    save_data(df, output_path)


if __name__ == "__main__":
    main()
