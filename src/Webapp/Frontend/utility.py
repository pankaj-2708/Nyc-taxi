import requests

base_url="http://127.0.0.1:8000"
def get_trip_duration(params_):
    return requests.request(
        url=f"{base_url}/trip_duration", method="get", json=params_
    ).json()["trip_duration_in_seconds"]

