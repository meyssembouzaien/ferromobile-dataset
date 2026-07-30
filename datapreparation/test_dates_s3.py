import requests
import time

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
lat, lon = 45.3974, 3.6556  # cluster central de la ligne

candidates = [
    "2023-06-14", "2023-06-30", "2023-07-15",
    "2023-07-24", "2023-08-10", "2022-06-18",
    "2022-07-20", "2022-08-02", "2021-07-14",
    "2021-08-11",
]

print(f"{'Date':<15} {'Max mm/h':>10} {'Moy mm/h':>10}")
print("-" * 38)

for date in candidates:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": date,
        "end_date":   date,
        "hourly":     "precipitation",
        "timezone":   "Europe/Paris",
        "models":     "era5",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    vals = r.json().get("hourly", {}).get("precipitation", [])
    vals = [v for v in vals if v is not None]
    mx   = max(vals) if vals else 0
    avg  = sum(vals) / len(vals) if vals else 0
    flag = " ✓" if mx >= 2.0 else ""
    print(f"{date:<15} {mx:>10.3f} {avg:>10.3f}{flag}")
    time.sleep(1.5)
