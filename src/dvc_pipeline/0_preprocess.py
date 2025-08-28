from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def load_data(input_path):
    return pd.read_csv(input_path)


def save_data(df, output_path):
    df.to_csv(output_path / "data.csv", index=False)

def bearing_diff(row):
    lat1_rad = np.deg2rad(row["pickup_latitude"])
    lat2_rad = np.deg2rad(row["dropoff_latitude"])
    long1_rad = np.deg2rad(row["pickup_longitude"])
    long2_rad = np.deg2rad(row["dropoff_longitude"])    

    delta_long = long2_rad - long1_rad

    x = np.sin(delta_long) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_long)

    theta = np.arctan2(x, y)
    bearing = (np.degrees(theta) + 360) % 360
    return bearing
    
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

# using map api elevetaion diff, and road distance between two points can be added
def preprocess(df, dropoff_datetime):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_minute"] = df["pickup_datetime"].dt.minute
    df["pickup_second"] = df["pickup_datetime"].dt.second
    df["pickup_day_of_week"] = df["pickup_datetime"].dt.day_of_week

    # pickup is optional because pickup and dropoff are exactly same in almost all rows
    if dropoff_datetime:
        df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
        df["dropoff_month"] = df["dropoff_datetime"].dt.month
        df["dropoff_day"] = df["dropoff_datetime"].dt.day
        df["dropoff_day"] = df["dropoff_datetime"].dt.hour
        df["dropoff_minute"] = df["dropoff_minute"].dt.minute
        df["dropoff_second"] = df["dropoff_datetime"].dt.second
        df["dropoff_day_of_week"] = df["dropoff_datetime"].dt.day_of_week

    df["distance_in_km"] = df.apply(convert_to_dist, axis=1)
    df['lat_diff'] = df['pickup_latitude'] - df['dropoff_latitude']
    df['long_diff'] = df['pickup_longitude'] - df['dropoff_longitude']
    df["bearing"] = df.apply(bearing_diff, axis=1)
    df['mid_lat'] = df['pickup_latitude']/2 + df['dropoff_latitude']/2
    df['mid_long'] = df['pickup_longitude']/2 + df['dropoff_longitude']/2
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
    df = preprocess(df, params["dropoff_datetime"])

    save_data(df, output_path)


if __name__ == "__main__":
    main()
