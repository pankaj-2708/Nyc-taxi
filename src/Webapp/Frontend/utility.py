import requests
import dotenv
import os
from pathlib import Path

pth = Path(__file__)
home_dir = pth.parent.parent.parent.parent
dotenv.load_dotenv(home_dir / ".env")


def get_trip_duration(params_):
    return requests.request(
        url=f"{os.getenv("base_url")}/trip_duration", method="get", json=params_
    ).json()["trip_duration_in_seconds"]
