from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def load_data(input_path):
    return pd.read_csv(input_path)


def save_data(df, output_path):
    df.to_csv(output_path / "data.csv", index=False)


def convert_to_dist(row):
    lat1_rad = np.deg2rad(row["pickup_latitude"])
    lat2_rad = np.deg2rad(row["dropoff_latitude"])
    long1_rad = np.deg2rad(row["pickup_longitude"])
    long2_rad = np.deg2rad(row["dropoff_longitude"])
    return (
        2
        * 6371
        * np.arcsin(
            (
                np.sin((lat2_rad - lat1_rad) / 2) ** 2
                + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin((long2_rad - long1_rad) / 2) ** 2
            )
            ** 0.5
        )
    )


def preprocess(df, pickup_datetime):
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    df["dropoff_month"] = df["dropoff_datetime"].dt.month
    df["dropoff_day"] = df["dropoff_datetime"].dt.day
    df["dropoff_day_of_week"] = df["dropoff_datetime"].dt.day_of_week

    # pickup is optional because pickup and dropoff are exactly same in almost all rows
    if pickup_datetime:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df["pickup_month"] = df["pickup_datetime"].dt.month
        df["pickup_day"] = df["pickup_datetime"].dt.day
        df["pickup_day_of_week"] = df["pickup_datetime"].dt.day_of_week

    df["distance_in_km"] = df.apply(convert_to_dist, axis=1)
    df.drop(columns=["pickup_datetime", "dropoff_datetime", "id"], inplace=True)
    return df


def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    input_path = home_dir / "data" / "raw" / "dataset.csv"

    output_path = home_dir / "data" / "processed"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["preprocess"]

    df = load_data(input_path)
    df = preprocess(df, params["pickup_datetime"])

    save_data(df, output_path)


if __name__ == "__main__":
    main()
