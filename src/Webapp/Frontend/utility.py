import requests

base_url="13.53.39.225:8000"
def get_trip_duration(params_):
    return requests.request(
        url=f"{base_url}/trip_duration", method="get", json=params_
    ).json()["trip_duration_in_seconds"]

