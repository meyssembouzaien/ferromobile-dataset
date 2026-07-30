"""
prepare_data.py — Étape 2 : Prétraitement et calcul de couverture réseau
==========================================================================
Entrée  : data/raw/{ligne_gps.csv, antennes_anfr.csv, tunnels.csv}
Sortie  : data/processed/dataset_base.csv

Path loss model: ITU-R P.1812 (crc-covlib) only. No analytical fallback.
If the library fails to load, the script exits — it does not silently
degrade to a less accurate model.
"""

import math
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import pyproj
import geopandas as gpd
import shapely.ops
from shapely.geometry import LineString
from scipy.spatial import cKDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
INPUT_DIR  = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CRC_COVLIB_PATH  = "/home/meyssem/Documents/ferromobile_project/crc-covlib/python-wrapper"
TERRAIN_DATA_DIR = os.path.join(INPUT_DIR, "terrain")
LANDCOVER_DIR    = os.path.join(INPUT_DIR, "landcover")

TOPO_URL   = "https://api.opentopodata.org/v1/eudem25m"
TOPO_BATCH = 100

_CRC_AVAILABLE = False
covlib = None
_CrcSim = None

if CRC_COVLIB_PATH not in sys.path:
    sys.path.insert(0, CRC_COVLIB_PATH)
try:
    import importlib
    covlib  = importlib.import_module("crc_covlib.simulation")
    _CrcSim = getattr(covlib, "Simulation", None)
    if _CrcSim and getattr(covlib, "_covlib_cdll", None):
        _t = _CrcSim()
        _t.SetTransmitterLocation(45.4, 3.73)
        _t.SetTransmitterHeight(35.0)
        _t.SetTransmitterFrequency(900.0)
        _t.SetReceiverHeightAboveGround(4.0)
        _t.SetPropagationModel(covlib.PropagationModel.ITU_R_P_1812)
        _pl = _t.GenerateReceptionPointResult(45.38, 3.71)
        if _pl and not math.isnan(float(_pl)) and 50 < float(_pl) < 280:
            print(f"✓ crc-covlib P.1812 OK (path loss test = {float(_pl):.1f} dB)")
            _CRC_AVAILABLE = True
        else:
            print(f"✗ crc-covlib chargé mais test P.1812 anormal ({_pl})")
    else:
        print("✗ crc-covlib : lib C++ non chargée")
except Exception as e:
    print(f"✗ crc-covlib : {e}")

if not _CRC_AVAILABLE:
    print("\n✗ ERREUR FATALE : crc-covlib indisponible. P.1812 est requis, "
          "aucun modèle de repli n'est utilisé.")
    sys.exit(1)

_TERRAIN_AVAILABLE   = (os.path.isdir(TERRAIN_DATA_DIR)
                        and bool(os.listdir(TERRAIN_DATA_DIR)))
_LANDCOVER_AVAILABLE = (os.path.isdir(LANDCOVER_DIR)
                        and bool(os.listdir(LANDCOVER_DIR)))

_P1812_CLUTTER = {}
_TERR_ELEV_SRTM3 = None
try:
    _P1812_CLUTTER = {
        "plaine":       covlib.P1812ClutterCategory.P1812_OPEN_RURAL,
        "foret_legere": covlib.P1812ClutterCategory.P1812_URBAN_TREES_FOREST,
        "foret_dense":  covlib.P1812ClutterCategory.P1812_URBAN_TREES_FOREST,
        "urbain":       covlib.P1812ClutterCategory.P1812_SUBURBAN,
    }
except AttributeError:
    _P1812_CLUTTER = {"plaine": 2, "foret_legere": 4, "foret_dense": 4, "urbain": 3}
for _cand in ["TerrainElevDataSource", "TerrainElevSource", "ElevDataSource"]:
    _ec = getattr(covlib, _cand, None)
    if _ec:
        for _m in _ec:
            if "SRTM" in _m.name.upper():
                _TERR_ELEV_SRTM3 = _m; break
    if _TERR_ELEV_SRTM3:
        break

TRONCON_KM_MAX = 40.0

TECH_PROFILES = {
    ("2G",   900): (20,   650, 1000, 0.24,  0.05, 3.0,  1e-2, -0.21, "lin"),
    ("3G",   900): (18.5, 120, 160,  1.3,   1.0,  2.0,  1e-3, -0.20, "lin"),
    ("3G",  2100): (7,    120, 160,  1.3,   1.0,  2.0,  1e-3, -0.20, "lin"),
    ("4G",   700): (15,   30,  80,  13.2,   1.0,  0.5,  5e-4, -0.05, "lin"),
    ("4G",   800): (17,   30,  80,  42.0,   1.0,  0.5,  5e-4, -0.05, "lin"),
    ("4G",  1800): (7,    30,  80,  35.0,   1.0,  0.5,  5e-4, -0.05, "lin"),
    ("4G",  2100): (5,    30,  80,  28.0,   1.0,  0.5,  5e-4, -0.05, "lin"),
    ("4G",  2600): (4,    30,  80,  50.0,   1.0,  0.5,  5e-4, -0.05, "lin"),
    ("5G",   700): (10,   15,  35, 80.0,   3.0,  0.6,  1e-4, -0.06, "lin"),
    ("5G",  2100): (5,    15,  35, 150.0, 5.0,  0.6,  1e-4, -0.06, "sig_soft"),
    ("5G",  3500): (2,    15,  35, 365.0, 10.0, 0.6,  1e-4, -0.06, "sig_hard"),
}
DEFAULT_PROFILE = (5, 50, 150, 10.0, 0.5, 1.0, 1e-4, -0.10, "lin")

MAT_HEIGHT_M = {"2G": 40.0, "3G": 38.0, "4G": 35.0, "5G": 25.0}
P_TX_DBM = {"2G": 43, "3G": 43, "4G": 46, "5G": 49}
BW_NOISE_MHZ = {"2G": 0.2, "3G": 5.0, "4G": 10.0, "5G": 20.0}
BW_SHANNON_MHZ = {"2G": 0.2, "3G": 5.0, "4G": 10.0, "5G": 100.0}
NOISE_FIGURE_DB = 7.0
SNR_MAX_DB_PER_TECH = {"2G": 22.0, "3G": 28.0, "4G": 32.0, "5G": 35.0}
SNR_MAX_DB = 35.0
SHANNON_EFF = 0.7
BITS_PER_PKT = 400

CODING_GAIN_DB = {"2G": 2.0, "3G": 4.0, "4G": 6.0, "5G": 8.0}

DEBIT_MIN_MBPS = 5.0
SNR_MIN_DB     = 5.0
RTT_MAX_MS     = 500.0

TUNNEL_BUFFER_M = 50

LOS_SAMPLE_PER_KM = 12
LOS_MAX_SAMPLES   = 400
LOS_K_FACTOR      = 4 / 3
LOS_RE_EFFECTIVE  = 6_371_000.0 * LOS_K_FACTOR

CORRIDOR_RAYON_KM = 10.0

TOP_K_PAR_GENERATION = {"2G": 3, "3G": 6, "4G": 8, "5G": 8}

_ELEV_CACHE      = {}
_ELEV_CACHE_FILE = os.path.join(BASE_DIR, "data", "cache", "elev_cache.json")
_RNG_BER   = np.random.default_rng(42)
_RNG_DELTA = np.random.default_rng(123)
_RNG_VIT   = np.random.default_rng(456)

def _load_elev_cache():
    import json as _j
    if os.path.exists(_ELEV_CACHE_FILE):
        try:
            with open(_ELEV_CACHE_FILE) as f:
                raw = _j.load(f)
            for k, v in raw.items():
                lat, lon = map(float, k.split(","))
                _ELEV_CACHE[(lat, lon)] = v
            print(f"  ✓ Cache altitudes : {len(_ELEV_CACHE)} points")
        except Exception as e:
            print(f"  ⚠ Cache illisible : {e}")

def _save_elev_cache():
    import json as _j
    os.makedirs(os.path.dirname(_ELEV_CACHE_FILE), exist_ok=True)
    raw = {f"{lat},{lon}": v for (lat, lon), v in _ELEV_CACHE.items()}
    with open(_ELEV_CACHE_FILE, "w") as f:
        _j.dump(raw, f)

_load_elev_cache()

_dem_dataset   = None
_dem_transform = None
_dem_checked   = False

def _init_dem():
    global _dem_dataset, _dem_transform, _dem_checked
    if _dem_dataset is not None:
        return True
    if _dem_checked:
        return False
    _dem_checked = True
    if not os.path.isdir(TERRAIN_DATA_DIR):
        return False
    try:
        import rasterio as _rio
        expected = os.path.join(TERRAIN_DATA_DIR, "eu_dem_courpiere_ambert.tif")
        if os.path.isfile(expected):
            path = expected
        else:
            exts = (".tif", ".tiff", ".hgt", ".bil", ".asc")
            rasters = [f for f in os.listdir(TERRAIN_DATA_DIR) if f.lower().endswith(exts)]
            if not rasters:
                return False
            path = os.path.join(TERRAIN_DATA_DIR, sorted(rasters)[0])
            print(f"  ⚠ eu_dem_courpiere_ambert.tif absent, fallback sur {os.path.basename(path)} "
                  f"(vérifier que ce raster couvre bien la zone d'étude)")
        _dem_dataset  = _rio.open(path)
        _dem_transform = _dem_dataset.transform
        print(f"  ✓ DEM local chargé : {os.path.basename(path)}")
        return True
    except Exception as e:
        print(f"  ⚠ Échec ouverture DEM local ({e})")
        return False

def _get_elevations_dem_batch(lats, lons):
    import requests as _req

    n = len(lats)
    result = [None] * n

    missing_idx = []
    for i, (la, lo) in enumerate(zip(lats, lons)):
        key = (round(la, 4), round(lo, 4))
        if key in _ELEV_CACHE:
            result[i] = _ELEV_CACHE[key]
        else:
            missing_idx.append(i)

    if not missing_idx:
        return result

    if _init_dem():
        try:
            coords = [(lons[i], lats[i]) for i in missing_idx]
            nodata = _dem_dataset.nodata
            still_missing = []
            for j, val in enumerate(_dem_dataset.sample(coords)):
                v = float(val[0])
                idx = missing_idx[j]
                if nodata is not None and v == nodata:
                    still_missing.append(idx)
                else:
                    result[idx] = v
                    key = (round(lats[idx], 4), round(lons[idx], 4))
                    _ELEV_CACHE[key] = v
            missing_idx = still_missing
        except Exception:
            pass

    if not missing_idx:
        return result

    for start in range(0, len(missing_idx), TOPO_BATCH):
        batch_idx = missing_idx[start : start + TOPO_BATCH]
        locations_str = "|".join(f"{lats[i]},{lons[i]}" for i in batch_idx)
        try:
            r = _req.get(
                TOPO_URL,
                params={"locations": locations_str, "interpolation": "bilinear"},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "OK":
                raise ValueError(f"API status: {data.get('status')}")
            for j, res in enumerate(data.get("results", [])):
                idx = batch_idx[j]
                v = res.get("elevation")
                result[idx] = v
                if v is not None:
                    key = (round(lats[idx], 4), round(lons[idx], 4))
                    _ELEV_CACHE[key] = v
        except Exception as e:
            print(f"\n  ⚠ OpenTopoData batch échoué ({e}) — None pour {len(batch_idx)} pts")

        if start + TOPO_BATCH < len(missing_idx):
            time.sleep(1.1)

    return result

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))

def haversine_km_vec(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlmb = np.radians(lon2_arr - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def normalize_generation(gen_raw):
    g = str(gen_raw).strip().upper()
    if g in ("GSM", "2G", "EDGE"):         return "2G"
    if g in ("UMTS", "3G", "HSPA"):        return "3G"
    if g in ("LTE", "4G", "LTE-A"):        return "4G"
    if g in ("NR", "5G", "5GNR", "NR5G"): return "5G"
    return g

def get_profile(generation, bande_mhz):
    try:
        bande = int(float(bande_mhz))
    except (TypeError, ValueError):
        return DEFAULT_PROFILE
    return TECH_PROFILES.get((normalize_generation(generation), bande), DEFAULT_PROFILE)

def get_portee_km(generation, bande_mhz):
    return get_profile(generation, bande_mhz)[0]

def _get_elevation(lat, lon, df_fallback):
    key = (round(lat, 4), round(lon, 4))
    if key in _ELEV_CACHE:
        return _ELEV_CACHE[key]
    vals = _get_elevations_dem_batch([lat], [lon])
    if vals[0] is not None:
        return vals[0]
    dists = (df_fallback["lat"] - lat)**2 + (df_fallback["lon"] - lon)**2
    alt = float(df_fallback.loc[dists.idxmin(), "altitude_m"])
    _ELEV_CACHE[key] = alt
    return alt

_ESA_TO_VEG = {
    10: "foret_dense", 20: "foret_legere", 30: "plaine",  40: "plaine",
    50: "urbain",      60: "plaine",       70: "plaine",  80: "plaine",
    90: "plaine",      95: "foret_legere", 100: "plaine"
}
_wc_dataset = _wc_transform = None
_veg_cache: dict = {}

def _init_worldcover():
    global _wc_dataset, _wc_transform
    if _wc_dataset is not None:
        return True
    if not os.path.isdir(LANDCOVER_DIR):
        return False
    try:
        import rasterio as _rio
        tifs = [f for f in os.listdir(LANDCOVER_DIR)
                if f.lower().endswith((".tif", ".tiff"))]
        if not tifs:
            return False
        path = os.path.join(LANDCOVER_DIR, sorted(tifs)[0])
        _wc_dataset = _rio.open(path)
        _wc_transform = _wc_dataset.transform
        return True
    except Exception:
        return False

def get_vegetation(lat, lon, distance_km=None):
    key = (round(lat, 4), round(lon, 4))
    if key in _veg_cache:
        return _veg_cache[key]
    veg = None
    if _init_worldcover():
        try:
            import rasterio as _rio
            row, col = _rio.transform.rowcol(_wc_transform, lon, lat)
            val = int(_wc_dataset.read(1, window=_rio.windows.Window(col, row, 1, 1))[0, 0])
            veg = _ESA_TO_VEG.get(val)
        except Exception:
            pass
    if veg is None:
        d = distance_km or 0.0
        if d <= 3.0:    veg = "urbain"
        elif d <= 8.0:  veg = "plaine"
        elif d <= 22.0: veg = "foret_dense"
        elif d <= 27.0: veg = "foret_legere"
        else:           veg = "urbain"
    _veg_cache[key] = veg
    return veg

def resample_ligne(df_ligne, step_m=20):
    lats = df_ligne["lat"].values
    lons = df_ligne["lon"].values
    alts = df_ligne["altitude_m"].values
    seg_km = haversine_km_vec(lats[:-1], lons[:-1], lats[1:], lons[1:])
    cum = np.concatenate(([0.0], np.cumsum(seg_km)))
    total_km = cum[-1]
    targets  = np.linspace(0, total_km, int(total_km / (step_m / 1000)) + 1)
    return pd.DataFrame({
        "point_id":    range(len(targets)),
        "lat":         np.round(np.interp(targets, cum, lats), 6),
        "lon":         np.round(np.interp(targets, cum, lons), 6),
        "altitude_m":  np.round(np.interp(targets, cum, alts), 1),
        "distance_km": np.round(targets, 4),
    })

def mark_tunnels(df_ligne, df_tunnels):
    lats = df_ligne["lat"].values
    lons = df_ligne["lon"].values
    seg_km   = haversine_km_vec(lats[:-1], lons[:-1], lats[1:], lons[1:])
    cum_dist = np.concatenate(([0.0], np.cumsum(seg_km)))

    coords_np = np.column_stack([lats, lons])
    tree      = cKDTree(coords_np)
    in_tunnel = pd.Series(False, index=df_ligne.index)
    buf_km    = TUNNEL_BUFFER_M / 1000.0
    has_sortie = ("lat_sortie" in df_tunnels.columns and
                  "lon_sortie" in df_tunnels.columns)

    for _, t in df_tunnels.iterrows():
        lat_e = float(t["lat_entree"]); lon_e = float(t["lon_entree"])
        _, idx_e = tree.query([lat_e, lon_e])
        d_e = cum_dist[idx_e]
        if has_sortie:
            lat_s = float(t["lat_sortie"]); lon_s = float(t["lon_sortie"])
            _, idx_s = tree.query([lat_s, lon_s])
            d_s = cum_dist[idx_s]
        else:
            d_s = d_e + float(t.get("longueur_m", 200)) / 1000.0
        d_lo = min(d_e, d_s) - buf_km
        d_hi = max(d_e, d_s) + buf_km
        in_tunnel |= (cum_dist >= d_lo) & (cum_dist <= d_hi)

    return in_tunnel

def check_los_binaire(pt_lat, pt_lon, pt_alt,
                      ant_lat, ant_lon, ant_alt_abs,
                      df_elev,
                      ant_height_m=30.0):
    dist_km = haversine_km(pt_lat, pt_lon, ant_lat, ant_lon)
    if dist_km < 0.05:
        return True

    num_points = int(min(
        max(16, dist_km * LOS_SAMPLE_PER_KM) + 1,
        LOS_MAX_SAMPLES
    ))

    lats = np.linspace(ant_lat, pt_lat, num_points)
    lons = np.linspace(ant_lon, pt_lon, num_points)

    raw_elev = _get_elevations_dem_batch(lats.tolist(), lons.tolist())
    elevations = np.array(raw_elev, dtype=object)
    missing = [i for i, v in enumerate(elevations) if v is None]
    if missing:
        for i in missing:
            elevations[i] = _get_elevation(lats[i], lons[i], df_elev)
    elevations = elevations.astype(float)

    if np.isnan(elevations).any():
        nans = np.isnan(elevations)
        if np.all(nans):
            elevations[:] = 0.0
        else:
            x = np.arange(num_points)
            elevations[nans] = np.interp(x[nans], x[~nans], elevations[~nans])
            elevations[np.isnan(elevations)] = 0.0

    ant_elev  = float(elevations[0]) + ant_height_m
    traj_elev = float(elevations[-1])
    los_heights = np.linspace(ant_elev, traj_elev, num_points)

    Dm = dist_km * 1000.0
    s  = np.linspace(0.0, Dm, num_points)
    curvature = (s * (Dm - s)) / (2.0 * LOS_RE_EFFECTIVE)

    terrain_effective = elevations + curvature

    if num_points > 2:
        los_clear = np.all(terrain_effective[1:-1] <= los_heights[1:-1])
    else:
        los_clear = True

    return bool(los_clear)

def _path_loss_p1812(pt_lat, pt_lon, ant_lat, ant_lon,
                     ant_height_agl_m, freq_mhz, vegetation):
    try:
        sim = _CrcSim()
        sim.SetTransmitterLocation(ant_lat, ant_lon)
        sim.SetTransmitterHeight(ant_height_agl_m)
        sim.SetTransmitterFrequency(freq_mhz)
        sim.SetReceiverHeightAboveGround(4.0)
        if _TERR_ELEV_SRTM3:
            try:
                sim.SetPrimaryTerrainElevDataSource(_TERR_ELEV_SRTM3)
                if _TERRAIN_AVAILABLE:
                    sim.SetTerrainElevDataSourceDirectory(TERRAIN_DATA_DIR)
            except Exception:
                pass
        clutter = _P1812_CLUTTER.get(vegetation)
        if clutter is not None:
            try:
                sim.SetDefaultLandCoverClassMapping(clutter)
            except Exception:
                pass
        sim.SetPropagationModel(covlib.PropagationModel.ITU_R_P_1812)
        try:
            sim.SetITURP1812TimePercentage(50)
            sim.SetITURP1812LocationPercentage(50)
        except Exception:
            pass
        path_loss = sim.GenerateReceptionPointResult(pt_lat, pt_lon)
        if path_loss is None:
            return None
        pl = float(path_loss)
        if math.isnan(pl) or math.isinf(pl) or pl <= 50.0 or pl > 280.0:
            return None
        return round(pl, 2)
    except Exception:
        return None

def snr_from_path_loss(path_loss_db, gen_norm):
    bw    = BW_NOISE_MHZ.get(gen_norm, 5.0)
    noise = -174 + 10*math.log10(bw*1e6) + NOISE_FIGURE_DB
    snr_raw = P_TX_DBM.get(gen_norm, 43) - path_loss_db - noise
    return round(min(snr_raw, SNR_MAX_DB_PER_TECH.get(gen_norm, SNR_MAX_DB)), 2)

def _erfc_approx(x):
    t = 1.0 / (1.0 + 0.3275911*abs(x))
    return (0.254829592*t - 0.284496736*t**2 + 1.421413741*t**3
            - 1.453152027*t**4 + 1.061405429*t**5) * math.exp(-x*x)

def ber_physique(snr_db, gen_norm, rng=None):
    if rng is None:
        rng = _RNG_BER
    snr_lin = 10 ** (max(-10.0, min(50.0, snr_db)) / 10)
    if gen_norm == "2G":
        ber = 0.5    * _erfc_approx(math.sqrt(max(0.0, 0.68*snr_lin)))
    elif gen_norm == "3G":
        ber = 0.5    * _erfc_approx(math.sqrt(max(0.0, snr_lin/2)))
    elif gen_norm == "4G":
        ber = (3/8)  * _erfc_approx(math.sqrt(max(0.0, snr_lin/10)))
    else:
        ber = (7/24) * _erfc_approx(math.sqrt(max(0.0, snr_lin/42)))
    return round(max(1e-6, ber * (10 ** rng.normal(0.0, 0.3))), 12)

SIGMOID_PARAMS = {
    "sig_soft": (0.896, 2.499),
    "sig_hard": (2.331, 1.017),
}

def calc_debit_empirique(dist_km, portee_km, debit_100m, debit_min, profil):
    d = max(dist_km, 0.0)
    if profil in SIGMOID_PARAMS:
        a, d0 = SIGMOID_PARAMS[profil]
        debit = debit_min + (debit_100m - debit_min) / (1 + math.exp(a * (d - d0)))
    else:
        debit = debit_100m - (debit_100m - debit_min) * d / portee_km
    return round(max(debit_min, debit), 4)

def calc_rtt_empirique(dist_km, portee_km, rtt_min, rtt_max):
    ratio = min(1.0, max(0.0, dist_km / portee_km))
    return round(rtt_min + ratio * (rtt_max - rtt_min), 1)

def calc_metrics(dist_km, generation, bande_mhz, vegetation, los_category,
                 vitesse_kmh=80.0, pt_lat=None, pt_lon=None,
                 ant_lat=None, ant_lon=None, ant_mat_m=None,
                 modele_debit="empirique"):
    prof = get_profile(generation, bande_mhz)
    portee_km, rtt_min, rtt_max, debit_max, debit_min, impact_vitesse, _, _, profil_debit = prof

    try:
        freq_mhz = max(150, int(float(bande_mhz)))
    except Exception:
        freq_mhz = 900

    d        = max(dist_km, 0.05)
    gen_norm = normalize_generation(generation)

    if los_category == 0:
        return (None, float("nan"), -99.0, 0.0,
                float("nan"), float("nan"), 1.0, 1.05,
                impact_vitesse, 0.0, "TUNNEL")

    perte_db = _path_loss_p1812(pt_lat, pt_lon, ant_lat, ant_lon,
                                ant_mat_m, freq_mhz, vegetation)
    if perte_db is None:
        return (None, float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), "P1812_FAILED")

    snr_base     = snr_from_path_loss(perte_db, gen_norm)
    snr_adjusted = snr_base

    # ── [CORRIGÉ] Effet vitesse : la vitesse DEGRADE le SNR (effet Doppler,
    # ICI inter-porteuses, erreurs d'estimation canal). Le terme est soustrait,
    # pas ajouté. Le cap par génération est réappliqué après pour éviter de
    # dépasser le plafond physique du récepteur.
    # Ref : impact_vitesse est plus élevé pour 2G (3.0 dB) que 5G (0.6 dB)
    # car les sous-porteuses OFDM larges (4G/5G) tolèrent mieux le Doppler
    # que les modulations à bande étroite (GMSK/QPSK).
    if vitesse_kmh > 50:
        snr_adjusted = round(snr_adjusted - impact_vitesse * (vitesse_kmh / 100.0), 2)
        snr_adjusted = min(snr_adjusted, SNR_MAX_DB_PER_TECH.get(gen_norm, SNR_MAX_DB))

    if modele_debit == "empirique":
        debit_adj = calc_debit_empirique(d, portee_km, debit_max, debit_min, profil_debit)
        rtt       = calc_rtt_empirique(d, portee_km, rtt_min, rtt_max)
    else:
        bw_eff    = BW_SHANNON_MHZ.get(gen_norm, 5.0)
        snr_lin   = max(0.001, 10 ** (max(-10.0, snr_adjusted) / 10))
        ratio     = max(0.0, min(1.0, 1.0 - d / portee_km))
        debit_nom = round(debit_min + ratio * (debit_max - debit_min), 4)
        debit_adj = round(min(bw_eff * math.log2(1 + snr_lin) * SHANNON_EFF,
                              debit_max, debit_nom + 0.01), 4)
        rtt = round(rtt_min + (1 - ratio) * (rtt_max - rtt_min), 1)

    ber       = ber_physique(snr_adjusted, gen_norm, rng=_RNG_BER)
    delta_ber = round(ber * 1.05, 12)
    ber_eff   = ber_physique(snr_adjusted + CODING_GAIN_DB.get(gen_norm, 0.0),
                              gen_norm, rng=_RNG_BER)
    log_pc    = BITS_PER_PKT * math.log(max(1e-15, 1.0 - ber_eff))
    pkt_loss  = round(min(100.0, max(0.0, (1.0 - math.exp(log_pc)) * 100.0)), 6)

    _sigma    = {"2G": 1.2, "3G": 1.5, "4G": 2.0, "5G": 2.5}.get(gen_norm, 1.5)
    delta_snr = round(max(-6.0, min(6.0, float(_RNG_DELTA.normal(0.0, _sigma*0.6)))), 3)

    direction     = 1.0 if _RNG_VIT.random() > 0.55 else -1.0
    delta_vitesse = round(direction * abs(impact_vitesse) * (vitesse_kmh / 100.0), 4)

    return (perte_db, snr_base, snr_adjusted, debit_adj,
            rtt, pkt_loss, ber, delta_ber, delta_vitesse, delta_snr, "P1812")

def creer_tampon(line_wgs84, rayon_km):
    to_m   = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    to_wgs = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    line_m = shapely.ops.transform(lambda x, y: to_m.transform(x, y), line_wgs84)
    buf_m  = line_m.buffer(rayon_km * 1000)
    return shapely.ops.transform(lambda x, y: to_wgs.transform(x, y), buf_m)

def build_point_antenna_association(df_ligne, df_antennes, corridor_km=CORRIDOR_RAYON_KM):
    # Rayon effectif = max(corridor, portée max de l'antenne)
    # Ainsi une antenne 2G/900 (portée 20km) peut couvrir même si à 15km du trajet
    rayon_max_ant = df_antennes["portee_km"].max()
    rayon_effectif = max(corridor_km, rayon_max_ant)

    line = LineString(df_ligne[["lon", "lat"]].to_numpy())
    zone_tampon = creer_tampon(line, rayon_effectif)   # au lieu de corridor_km
    ant_gdf = gpd.GeoDataFrame(
        df_antennes, geometry=gpd.points_from_xy(df_antennes["lon"], df_antennes["lat"]),
        crs="EPSG:4326")
    antennes_proches = ant_gdf[ant_gdf.intersects(zone_tampon)]
    print(f"  Tampon {rayon_effectif}km sur la ligne : {len(antennes_proches)}/{len(df_antennes)} antennes")

    # ... le reste ne change pas, le filtre fin par portée continue de faire son travail

    traj_xy   = df_ligne[["lon", "lat"]].to_numpy(dtype=float)
    point_ids = df_ligne["point_id"].to_numpy()
    tree      = cKDTree(traj_xy)

    rows = []
    for ant_idx, a in antennes_proches.iterrows():
        lon_a, lat_a, portee_km = float(a["lon"]), float(a["lat"]), float(a["portee_km"])
        rayon_deg = portee_km / 111.0
        idxs = tree.query_ball_point([lon_a, lat_a], rayon_deg)
        if not idxs:
            continue
        idxs  = np.asarray(idxs)
        dists = haversine_km_vec(lat_a, lon_a, traj_xy[idxs, 1], traj_xy[idxs, 0])
        mask  = dists <= portee_km
        for pid, d in zip(point_ids[idxs[mask]], dists[mask]):
            rows.append((pid, ant_idx, d))

    assoc = pd.DataFrame(rows, columns=["point_id", "ant_idx", "_dist"])
    if assoc.empty:
        return assoc

    assoc = assoc.merge(df_antennes[["generation"]], left_on="ant_idx", right_index=True)
    assoc = (assoc.sort_values("_dist")
                   .groupby(["point_id", "generation"])
                   .apply(lambda g: g.head(TOP_K_PAR_GENERATION.get(g.name[1], 8)),
                          include_groups=False))
    assoc = assoc.reset_index(level=["point_id", "generation"])
    return assoc.sort_values(["point_id", "_dist"]).drop(columns="generation")

def main(step_m=20, corridor_rayon_km=CORRIDOR_RAYON_KM, vitesse_kmh=80.0,
        modele_debit="empirique"):
    print("=" * 70)
    print(f"ÉTAPE 2 — prepare_data.py | Modèle path loss : P.1812 (crc-covlib) | débit/RTT : {modele_debit}")
    print(f"  step={step_m}m | corridor={corridor_rayon_km}km | v={vitesse_kmh}km/h")
    print("=" * 70)

    if _init_dem():
        print(f"  ✓ Source altitude LOS : raster local ({TERRAIN_DATA_DIR})")
    else:
        print(f"  ⚠ Pas de raster local → fallback API OpenTopoData EU-DEM 25m")

    print("\n[1/8] Chargement...")
    df_ligne    = pd.read_csv(f"{INPUT_DIR}/ligne_gps.csv")
    df_antennes = pd.read_csv(f"{INPUT_DIR}/antennes_anfr.csv")
    df_tunnels  = pd.read_csv(f"{INPUT_DIR}/tunnels.csv")
    print(f"  Tracé : {len(df_ligne)} pts | Antennes : {len(df_antennes)} | Tunnels : {len(df_tunnels)}")

    for col in ["ant_id", "support_id"]:
        if col in df_antennes.columns:
            df_antennes[col] = df_antennes[col].astype(str)

    print("\n[2/8] Filtrage tronçon...")
    df_ligne = df_ligne[df_ligne["distance_km"] <= TRONCON_KM_MAX].copy().reset_index(drop=True)
    df_ligne["distance_km"] = (df_ligne["distance_km"] - df_ligne["distance_km"].min()).round(4)
    print(f"  {len(df_ligne)} points — {df_ligne['distance_km'].max():.1f} km")

    print(f"\n[3/8] Rééchantillonnage à {step_m} m...")
    df_ligne = resample_ligne(df_ligne, step_m=step_m)
    print(f"  {len(df_ligne)} points — alt [{df_ligne['altitude_m'].min():.0f}–{df_ligne['altitude_m'].max():.0f} m]")

    print("\n[4/8] Marquage tunnels (conservés, non exclus)...")
    df_ligne["in_tunnel"] = mark_tunnels(df_ligne, df_tunnels)
    n_tun = df_ligne["in_tunnel"].sum()
    print(f"  {n_tun} points en tunnel (~{n_tun*step_m/1000:.2f} km)")

    print("\n[5/8] Végétation (ESA WorldCover ou fallback distance)...")
    df_ligne["vegetation"] = df_ligne.apply(
        lambda r: get_vegetation(r["lat"], r["lon"], r["distance_km"]), axis=1)
    print(f"  {df_ligne['vegetation'].value_counts().to_dict()}")

    print("\n[6/8] Nettoyage antennes ANFR + portées bibliographiques...")
    df_antennes["generation"] = df_antennes["generation"].apply(normalize_generation)
    df_antennes = df_antennes[df_antennes["generation"].isin(["2G","3G","4G","5G"])].copy()
    for col in ["ant_id", "support_id"]:
        if col in df_antennes.columns:
            df_antennes[col] = df_antennes[col].astype(str)
    if "coordonnees" in df_antennes.columns:
        def parse_coord(s):
            try:
                parts = str(s).replace(",", " ").split()
                return float(parts[0]), float(parts[1])
            except Exception:
                return None, None
        df_antennes[["ant_lat","ant_lon"]] = df_antennes["coordonnees"].apply(
            lambda s: pd.Series(parse_coord(s)))
    df_antennes = df_antennes.dropna(subset=["ant_lat","ant_lon"]).copy()
    df_antennes = df_antennes.rename(columns={"ant_lat": "lat", "ant_lon": "lon"})

    df_antennes["portee_km"] = df_antennes.apply(
        lambda r: get_portee_km(r["generation"], r.get("bande_mhz", 900)), axis=1)
    print(f"  {len(df_antennes)} antennes | {df_antennes['generation'].value_counts().to_dict()}")
    print(f"  Portées biblio : {df_antennes.groupby('generation')['portee_km'].mean().round(1).to_dict()} km (moy.)")

    print(f"\n[7/8] LOS binaire + métriques radio (P.1812)...")
    df_elev = df_ligne[["lat","lon","altitude_m"]].copy()

    def get_ant_alt_sol(row):
        key = (round(row["lat"],4), round(row["lon"],4))
        cached = _ELEV_CACHE.get(key)
        if cached is not None:
            return cached
        return _get_elevation(row["lat"], row["lon"], df_elev)

    df_antennes["ant_alt_sol"]   = df_antennes.apply(get_ant_alt_sol, axis=1)
    df_antennes["ant_alt_abs_m"] = df_antennes.apply(
        lambda r: round(r["ant_alt_sol"] + MAT_HEIGHT_M.get(r["generation"], 35.0), 1), axis=1)

    print("\n  Association point-antenne (tampon corridor + KDTree-sur-trajet)...")
    assoc = build_point_antenna_association(df_ligne, df_antennes, corridor_km=corridor_rayon_km)
    assoc_par_point = {pid: g for pid, g in assoc.groupby("point_id")} if not assoc.empty else {}
    print(f"  {len(assoc)} paires (point, antenne) après plafond TOP_K_PAR_GENERATION")

    rows         = []
    model_counts = {}
    n_tunnel     = 0
    n_los        = 0
    n_nlos_elim  = 0
    n_p1812_fail = 0
    n_pts = len(df_ligne)

    for idx, pt in df_ligne.iterrows():
        if idx % 300 == 0 and idx > 0:
            print(f"  Point {idx:5d}/{n_pts} | LOS={n_los} NLOS_élim={n_nlos_elim} "
                  f"tunnel={n_tunnel} | P1812_échoué={n_p1812_fail}...")

        pt_lat, pt_lon = pt["lat"], pt["lon"]
        pt_alt, in_tun = pt["altitude_m"], pt["in_tunnel"]
        veg = pt["vegetation"]

        pt_assoc = assoc_par_point.get(pt["point_id"])
        if pt_assoc is None or pt_assoc.empty:
            continue
        cands = df_antennes.loc[pt_assoc["ant_idx"].values].copy()
        cands["_dist"] = pt_assoc["_dist"].values

        for _, ant in cands.iterrows():
            gen     = ant["generation"]
            bande   = ant.get("bande_mhz", 900)
            dist    = ant["_dist"]
            alt_abs = ant["ant_alt_abs_m"]
            mat_m   = MAT_HEIGHT_M.get(gen, 35.0)

            if in_tun:
                los_category = 0
                n_tunnel += 1
            else:
                los_ok = check_los_binaire(
                    pt_lat, pt_lon, pt_alt,
                    ant["lat"], ant["lon"], alt_abs,
                    df_elev,
                    ant_height_m=mat_m)
                if not los_ok:
                    n_nlos_elim += 1
                    continue
                los_category = 1
                n_los += 1

            (perte_db, snr_base, snr_adjusted, debit_adj,
             rtt, pkt_loss, ber, delta_ber, delta_vitesse, delta_snr,
             model_used) = calc_metrics(
                dist, gen, bande, veg, los_category, vitesse_kmh,
                pt_lat=pt_lat, pt_lon=pt_lon,
                ant_lat=ant["lat"], ant_lon=ant["lon"], ant_mat_m=mat_m,
                modele_debit=modele_debit)

            model_counts[model_used] = model_counts.get(model_used, 0) + 1

            if model_used == "P1812_FAILED":
                n_p1812_fail += 1
                n_los -= 1
                continue

            if not in_tun and snr_adjusted == -99.0:
                snr_adjusted = float("nan")
            flag_debit_nul = int((not in_tun) and (debit_adj == 0.0))

            rows.append({
                "point_id":        pt["point_id"],
                "row_id":          len(rows),
                "lat":             round(pt_lat, 6),
                "lon":             round(pt_lon, 6),
                "distance_km":     round(pt["distance_km"], 4),
                "altitude_m":      round(pt_alt, 1),
                "vegetation":      veg,
                "in_tunnel":       in_tun,
                "ant_id":          str(ant.get("ant_id", "")),
                "operateur":       ant.get("operateur", ""),
                "generation":      gen,
                "bande_mhz":       bande,
                "commune":         ant.get("commune", ""),
                "ant_lat":         round(ant["lat"], 6),
                "ant_lon":         round(ant["lon"], 6),
                "ant_alt_abs_m":   alt_abs,
                "dist_ant_km":     round(dist, 4),
                "los_category":    los_category,
                "perte_db":        perte_db,
                "snr_db":          snr_base,
                "snr_adjusted_db": snr_adjusted,
                "debit_adjusted":  debit_adj,
                "rtt_ms":          rtt,
                "ber":             ber,
                "delta_ber":       delta_ber,
                "delta_vitesse":   delta_vitesse,
                "delta_snr_db":    delta_snr,
                "packet_loss_pct": pkt_loss,
                "path_loss_model": model_used,
                "flag_debit_nul":  flag_debit_nul,
            })

    print("\n[8/8] Finalisation...")
    df_out = pd.DataFrame(rows)
    for col in ["ant_id"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str)

    pts_avec_lien = set(df_out["point_id"]) if len(df_out) > 0 else set()
    pts_orphelins = df_ligne[~df_ligne["point_id"].isin(pts_avec_lien)]
    if len(pts_orphelins) > 0:
        print(f"  ⚠ {len(pts_orphelins)} points sans aucun lien valide "
              f"(tous NLOS, P1812 échoué, ou aucun candidat) → zone blanche explicite")
        placeholder_rows = [{
            "point_id":        pt["point_id"],
            "row_id":          None,
            "lat":             round(pt["lat"], 6),
            "lon":             round(pt["lon"], 6),
            "distance_km":     round(pt["distance_km"], 4),
            "altitude_m":      round(pt["altitude_m"], 1),
            "vegetation":      pt["vegetation"],
            "in_tunnel":       pt["in_tunnel"],
            "ant_id": "", "operateur": "",
            "generation": "", "bande_mhz": None, "commune": "",
            "ant_lat": None, "ant_lon": None, "ant_alt_abs_m": None,
            "dist_ant_km": None, "los_category": None,
            "perte_db": None, "snr_db": None, "snr_adjusted_db": float("nan"),
            "debit_adjusted": 0.0, "rtt_ms": float("nan"),
            "ber": None, "delta_ber": None, "delta_vitesse": None, "delta_snr_db": None,
            "packet_loss_pct": None, "path_loss_model": "AUCUN_CANDIDAT",
            "flag_debit_nul": 1,
        } for _, pt in pts_orphelins.iterrows()]
        placeholder_df = pd.DataFrame(placeholder_rows)
        for col in df_out.columns:
            if col not in placeholder_df.columns:
                placeholder_df[col] = None
        placeholder_df = placeholder_df[df_out.columns].astype(df_out.dtypes.to_dict(), errors="ignore")
        df_out = pd.concat([df_out, placeholder_df], ignore_index=True)
        df_out["row_id"] = range(len(df_out))

    total_liens = n_los + n_nlos_elim + n_tunnel
    print(f"\n  ── Bilan filtrage ──")
    print(f"  Liens tunnel (conservés): {n_tunnel:6d}")
    print(f"  Liens LOS conservés     : {n_los:6d} ({100*n_los/max(1,total_liens):.1f}%)")
    print(f"  Liens NLOS éliminés     : {n_nlos_elim:6d} ({100*n_nlos_elim/max(1,total_liens):.1f}%)")
    print(f"  Liens P1812 échoués     : {n_p1812_fail:6d}")
    print(f"  Modèles radio           : {model_counts}")

    if len(df_out) > 0:
        df_out["qualite_ok"] = (
            (df_out["debit_adjusted"] >= DEBIT_MIN_MBPS) &
            (df_out["snr_adjusted_db"].notna()) &
            (df_out["snr_adjusted_db"] >= SNR_MIN_DB) &
            (df_out["rtt_ms"].notna()) &
            (df_out["rtt_ms"] <= RTT_MAX_MS) &
            (~df_out["in_tunnel"]))
        pts_qok = df_out.groupby("point_id")["qualite_ok"].any()
        df_out["zone_blanche"] = df_out["point_id"].map(
            lambda pid: int(not pts_qok.get(pid, False))).astype(int)
        df_out.loc[df_out["in_tunnel"], "zone_blanche"] = 1

        hors   = df_out[~df_out["in_tunnel"]]
        pts_zb = int((~pts_qok).sum())
        print(f"\n  Lignes dataset          : {len(df_out)}")
        print(f"  Points uniques          : {df_out['point_id'].nunique()} / {len(df_ligne)} du trajet")
        print(f"  Antennes uniques        : {df_out.loc[df_out['ant_id'] != '', 'ant_id'].nunique()}")
        print(f"  Points couverts         : {len(pts_qok)-pts_zb}/{len(pts_qok)}")
        print(f"  Zones blanches          : {pts_zb} ({100*pts_zb/max(1,len(pts_qok)):.1f}%, "
              f"dont {len(pts_orphelins)} sans aucun candidat)")
        if len(hors) > 0:
            hors_avec_ant = hors[hors["ant_id"] != ""]
            ant_par_pt = hors_avec_ant.groupby("point_id")["ant_id"].nunique()
            print(f"  Antennes/point (hors tunnel, hors orphelins) : moy={ant_par_pt.mean():.1f} "
                  f"min={ant_par_pt.min()} max={ant_par_pt.max()}")
            print(f"  Débit adj. moyen        : {hors_avec_ant['debit_adjusted'].mean():.2f} Mbps")
            print(f"  SNR adj. moyen          : {hors_avec_ant['snr_adjusted_db'].mean():.1f} dB")

    _save_elev_cache()

    GEN_NUM_MAP = {"2G": 1, "3G": 2, "4G": 3, "5G": 4}
    VEG_NUM_MAP = {"plaine": 0, "urbain": 1, "foret_legere": 2, "foret_dense": 3}
    df_out["gen_num"] = df_out["generation"].map(GEN_NUM_MAP).fillna(0).astype(int)
    df_out["veg_num"] = df_out["vegetation"].map(VEG_NUM_MAP).fillna(0).astype(int)

    assert df_out["row_id"].is_unique, "ERREUR : row_id non unique"
    cols = ["row_id", "point_id"] + [c for c in df_out.columns
                                     if c not in ("row_id", "point_id")]
    df_out = df_out[cols]

    out_path = f"{OUTPUT_DIR}/dataset_base.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n✓ {out_path} — {len(df_out)} lignes × {len(df_out.columns)} colonnes")
    print("→ Prochaine étape : python enrich_data.py --scenario ALL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-m",      type=int,   default=20)
    parser.add_argument("--corridor-km", type=float, default=CORRIDOR_RAYON_KM)
    parser.add_argument("--vitesse-kmh", type=float, default=80.0)
    parser.add_argument("--modele-debit", choices=["shannon", "empirique"], default="empirique")
    args = parser.parse_args()
    main(args.step_m, args.corridor_km, args.vitesse_kmh, args.modele_debit)