from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime
import pandas as pd
import numpy as np
import pickle


class check_data(BaseModel):
    vendor_id: int = Field(gt=0)
    pickup_datetime: datetime
    passenger_count: int = Field(gt=0)
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: Literal["Y", "N"]


def bearing_diff(row):
    lat1_rad = np.deg2rad(row["pickup_latitude"])
    lat2_rad = np.deg2rad(row["dropoff_latitude"])
    long1_rad = np.deg2rad(row["pickup_longitude"])
    long2_rad = np.deg2rad(row["dropoff_longitude"])

    delta_long = long2_rad - long1_rad

    x = np.sin(delta_long) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(
        delta_long
    )

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


def process_data(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_minute"] = df["pickup_datetime"].dt.minute
    df["pickup_second"] = df["pickup_datetime"].dt.second
    df["pickup_day_of_week"] = df["pickup_datetime"].dt.day_of_week
    df["distance_in_km"] = df.apply(convert_to_dist, axis=1)
    df["lat_diff"] = df["pickup_latitude"] - df["dropoff_latitude"]
    df["long_diff"] = df["pickup_longitude"] - df["dropoff_longitude"]
    df["bearing"] = df.apply(bearing_diff, axis=1)
    df["mid_lat"] = df["pickup_latitude"] / 2 + df["dropoff_latitude"] / 2
    df["mid_long"] = df["pickup_longitude"] / 2 + df["dropoff_longitude"] / 2
    df.drop(columns=["pickup_datetime"], inplace=True)
    return df
