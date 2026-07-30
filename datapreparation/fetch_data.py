"""
fetch_data.py — Étape 1 : Collecte des données statiques
=========================================================
Ligne ferroviaire 785000 (Courpière–Ambert, ~38.6 km)

Tracé GPS :
  Lu directement depuis data/raw/courpiere_ambert_resampled.csv
  (colonnes attendues : point_id, lat, lon, distance_km)
  Rééchantillonné à SPACING_M mètres.

Sources complémentaires :
  1. Passages à niveau — SNCF Open Data + fallback OSM Overpass
  2. Tunnels           — 12 tunnels embarqués (vérifiés)
  3. ANFR              — antennes mobiles 2G/3G/4G/5G (rayon 10 km)
  4. OpenTopoData EU-DEM 25 m — altitude pour chaque point du tracé
  5. Raster DEM local  — grille EU-DEM couvrant toute la zone d'étude,
     nécessaire pour le test LOS de prepare_data.py

Sorties dans data/raw/ :
  ligne_gps.csv        — points rééchantillonnés + altitude (m)
  passages_niveau.csv  — passages à niveau filtrés
  tunnels.csv          — 12 tunnels
  antennes_anfr.csv    — antennes mobiles
  terrain/eu_dem_courpiere_ambert.tif — raster DEM local pour le test LOS

Usage :
  pip install requests pandas numpy scipy rasterio
  python fetch_data.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────────────────────────────────────
# Dépendance rasterio requise pour le raster DEM (étape 6/6). On le vérifie
# ici, tout de suite, plutôt que de le découvrir après ~20 min de collecte
# quand build_local_dem_raster() y arrive enfin.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import rasterio  # noqa: F401
    from rasterio.transform import from_origin  # noqa: F401
    _RASTERIO_OK = True
except ImportError:
    _RASTERIO_OK = False
    print("✗ rasterio n'est pas installé (pip install rasterio).")
    print("  Le raster DEM local (étape 6/6, requis par prepare_data.py) ne")
    print("  pourra pas être généré. Installez rasterio avant de continuer.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
TERRAIN_DIR = RAW_DIR / "terrain"
LIGNE_CODE = "785000"
RAYON_KM   = 10.0
SPACING_M  = 100

SNCF_BASE = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets"
ANFR_URL  = "https://data.anfr.fr/d4c/api/records/1.0/search/"
TOPO_URL  = "https://api.opentopodata.org/v1/eudem25m"
TOPO_BATCH = 100

# Pas de la grille du raster DEM local (mètres). 100m est cohérent avec
# SPACING_M et suffisant pour le test LOS (échantillonné à 12 pts/km ≈ 83m).
DEM_GRID_SPACING_M = 100
# Marge de sécurité (km) ajoutée au buffer RAYON_KM pour la grille DEM.
DEM_BUFFER_MARGIN_KM = 2.0

# Un batch de requête OpenTopoData peut échouer ponctuellement (timeout,
# 429 rate-limit). On retente avant d'abandonner le batch en NaN.
DEM_BATCH_MAX_RETRIES = 3
DEM_BATCH_RETRY_DELAY_S = 5.0
# Si plus de cette fraction de la grille reste NaN après tous les essais,
# le raster est jugé inexploitable : on refuse de l'écrire plutôt que de
# livrer silencieusement un DEM presque entièrement interpolé au plus proche
# voisin (ce qui reproduirait exactement le bug que ce raster doit corriger).
DEM_MAX_NAN_FRACTION = 0.15

COURPIERE = (45.7651, 3.5402)
AMBERT    = (45.5420, 3.7353)
ANFR_DEPTS = ["063", "043"]

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
OVERPASS_HEADERS = {
    "User-Agent": "FerroMobile/2.0 (railway connectivity research; academic)",
    "Accept": "application/json",
}
OVERPASS_BBOX_PN = "(45.53,3.53,45.76,3.76)"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 12 tunnels vérifiés — Courpière–Ambert
# ─────────────────────────────────────────────────────────────────────────────

TUNNELS: list[dict] = [
    {"nom": "Tunnel de Sauviat",       "longueur_m": 302, "lat_entree": 45.717035,  "lon_entree": 3.5310529, "lat_sortie": 45.7143803, "lon_sortie": 3.5302504},
    {"nom": "Tunnel d'Archimbaud",     "longueur_m": 142, "lat_entree": 45.7083611, "lon_entree": 3.5378128, "lat_sortie": 45.7074477, "lon_sortie": 3.5390899},
    {"nom": "Tunnel de Cublas",        "longueur_m": 193, "lat_entree": 45.7042076, "lon_entree": 3.5475462, "lat_sortie": 45.7035976, "lon_sortie": 3.5498699},
    {"nom": "Tunnel des Graves",       "longueur_m": 319, "lat_entree": 45.7005016, "lon_entree": 3.5565818, "lat_sortie": 45.7000257, "lon_sortie": 3.5606289},
    {"nom": "Tunnel de Saint Gervais", "longueur_m": 124, "lat_entree": 45.6920718, "lon_entree": 3.5971847, "lat_sortie": 45.6918742, "lon_sortie": 3.5987594},
    {"nom": "Tunnel de Constancis",    "longueur_m":  46, "lat_entree": 45.6807812, "lon_entree": 3.6169128, "lat_sortie": 45.6807474, "lon_sortie": 3.6175025},
    {"nom": "Tunnel d'Olliergues",     "longueur_m": 117, "lat_entree": 45.6742838, "lon_entree": 3.6334474, "lat_sortie": 45.6737559, "lon_sortie": 3.6347442},
    {"nom": "Tunnel de Chalard",       "longueur_m": 241, "lat_entree": 45.6703068, "lon_entree": 3.6411955, "lat_sortie": 45.6684614, "lon_sortie": 3.6428202},
    {"nom": "Tunnel de Got",           "longueur_m":  61, "lat_entree": 45.6652648, "lon_entree": 3.6491866, "lat_sortie": 45.665103,  "lon_sortie": 3.6499399},
    {"nom": "Tunnel de Flouvat",       "longueur_m": 168, "lat_entree": 45.6475144, "lon_entree": 3.6748709, "lat_sortie": 45.6475156, "lon_sortie": 3.6770262},
    {"nom": "Tunnel du Châtelet",      "longueur_m": 586, "lat_entree": 45.6236563, "lon_entree": 3.7197255, "lat_sortie": 45.6184188, "lon_sortie": 3.7205174},
    {"nom": "Tunnel de Perrier",       "longueur_m": 238, "lat_entree": 45.6168064, "lon_entree": 3.723567,  "lat_sortie": 45.6153495, "lon_sortie": 3.7213251},
]

# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires géographiques
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance orthodromique en kilomètres (formule haversine)."""
    R = 6_371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def interpolate_line(
    coords: list[tuple[float, float]],
    spacing_m: int = 100,
) -> list[dict]:
    """Rééchantillonne une séquence (lat, lon) à espacement constant spacing_m."""
    if len(coords) < 2:
        return [{"point_id": 0, "lat": coords[0][0], "lon": coords[0][1], "distance_km": 0.0}]

    result: list[dict] = [{
        "point_id": 0,
        "lat": round(coords[0][0], 6),
        "lon": round(coords[0][1], 6),
        "distance_km": 0.0,
    }]
    carry_m = 0.0

    for i in range(1, len(coords)):
        lat0, lon0 = coords[i - 1]
        lat1, lon1 = coords[i]
        seg_m = haversine_km(lat0, lon0, lat1, lon1) * 1_000
        if seg_m < 1e-6:
            continue
        dlat, dlon = lat1 - lat0, lon1 - lon0
        pos = spacing_m - carry_m
        while pos <= seg_m:
            f = pos / seg_m
            lat_p = round(lat0 + f * dlat, 6)
            lon_p = round(lon0 + f * dlon, 6)
            prev = result[-1]
            d_km = prev["distance_km"] + haversine_km(prev["lat"], prev["lon"], lat_p, lon_p)
            result.append({
                "point_id": len(result),
                "lat": lat_p,
                "lon": lon_p,
                "distance_km": round(d_km, 4),
            })
            pos += spacing_m
        carry_m = seg_m - (pos - spacing_m)

    return result


def build_kdtree(pts: list[dict]) -> cKDTree:
    """Construit un KD-tree sur les coordonnées (lat, lon) des points du tracé."""
    coords = np.array([[p["lat"], p["lon"]] for p in pts])
    return cKDTree(coords)


def dist_to_line_km(lats: np.ndarray, lons: np.ndarray, tree: cKDTree) -> np.ndarray:
    """Distance minimale (km) de chaque point (lat, lon) au tracé, via KD-tree."""
    query_pts = np.column_stack([lats, lons])
    raw_dists, idx = tree.query(query_pts)
    nn_coords = tree.data[idx]
    return np.array([
        haversine_km(lats[i], lons[i], nn_coords[i, 0], nn_coords[i, 1])
        for i in range(len(lats))
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tracé GPS — lecture depuis CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_trace(csv_path: Path, spacing_m: int = SPACING_M) -> list[dict]:
    """Charge le tracé GPS depuis csv_path et le rééchantillonne."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {csv_path}\n"
            "Vérifiez que le CSV est dans data/raw/."
        )

    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        "Latitude": "lat",
        "Longitude": "lon",
        "LAT": "lat",
        "LON": "lon",
    })

    missing = {"lat", "lon"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {csv_path.name} : {missing}\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            f"Tracé insuffisant : seulement {len(df)} point(s) valide(s)."
        )

    coords = list(zip(df["lat"], df["lon"]))

    d_start = haversine_km(*coords[0], *COURPIERE)
    d_end   = haversine_km(*coords[-1], *COURPIERE)

    if d_end < d_start:
        coords = coords[::-1]
        print("  ✓ Tracé inversé → orientation Courpière→Ambert")

    raw_len_km = sum(
        haversine_km(coords[i][0], coords[i][1],
                     coords[i+1][0], coords[i+1][1])
        for i in range(len(coords) - 1)
    )

    print(f"  Tracé brut : {len(coords)} pts | {raw_len_km:.2f} km")

    if not (28.0 < raw_len_km < 50.0):
        print("  ⚠ Longueur hors plage attendue [28–50 km]")

    pts = interpolate_line(coords, spacing_m=spacing_m)
    print(f"  → {len(pts)} points rééchantillonnés à {spacing_m} m")

    return pts


# ─────────────────────────────────────────────────────────────────────────────
# 2. Passages à niveau — SNCF + fallback OSM Overpass
# ─────────────────────────────────────────────────────────────────────────────

def _sncf_get_all(dataset: str, where: str) -> list[dict]:
    """Pagine automatiquement l'API SNCF et retourne tous les enregistrements."""
    url = f"{SNCF_BASE}/{dataset}/records"
    records, offset, limit = [], 0, 100
    while True:
        r = requests.get(url, params={"limit": limit, "offset": offset, "where": where}, timeout=30)
        r.raise_for_status()
        data  = r.json()
        batch = data.get("results", [])
        if not batch:
            break
        records.extend(batch)
        offset += limit
        if offset >= data.get("total_count", 0):
            break
    return records


def _pn_from_osm() -> pd.DataFrame:
    """Récupère les passages à niveau via Overpass (fallback)."""
    query = f"""
[out:json][timeout:25];
(
  node["railway"="level_crossing"]{OVERPASS_BBOX_PN};
  node["railway"="crossing"]{OVERPASS_BBOX_PN};
);
out body;
"""
    for mirror in OVERPASS_MIRRORS:
        try:
            r = requests.post(mirror, data={"data": query},
                              headers=OVERPASS_HEADERS, timeout=35)
            r.raise_for_status()
            elements = r.json().get("elements", [])
            if not elements:
                continue
            rows = [
                {
                    "libelle": el.get("tags", {}).get("name",
                               el.get("tags", {}).get("ref", f"PN_OSM_{el.get('id', '')}")),
                    "lat":    el.get("lat"),
                    "lon":    el.get("lon"),
                    "type_pn": el.get("tags", {}).get("railway", "level_crossing"),
                }
                for el in elements
            ]
            df = (pd.DataFrame(rows)
                    .dropna(subset=["lat", "lon"])
                    .reset_index(drop=True))
            print(f"  ✓ {len(df)} passages à niveau récupérés via OSM ({mirror})")
            return df
        except requests.RequestException as e:
            print(f"    ✗ {mirror} : {e}")
        time.sleep(2)

    print("  ⚠  Tous les miroirs Overpass sont inaccessibles.")
    return pd.DataFrame(columns=["libelle", "lat", "lon", "type_pn"])


def fetch_passages_niveau(pts: list[dict]) -> pd.DataFrame:
    """Passages à niveau SNCF (ou OSM en fallback), filtrés à RAYON_KM de la ligne."""
    print("  Appel API SNCF — passages à niveau...")
    df = pd.DataFrame()
    try:
        records = _sncf_get_all("liste-des-passages-a-niveau",
                                f"code_ligne='{LIGNE_CODE}'")
        rows = [
            {
                "libelle": rec.get("libelle_long", rec.get("libelle", "")),
                "lat":     (rec.get("position_geographique") or {}).get("lat"),
                "lon":     (rec.get("position_geographique") or {}).get("lon"),
                "type_pn": rec.get("type_pn", ""),
            }
            for rec in records
        ]
        df = pd.DataFrame(rows).dropna(subset=["lat", "lon"]).reset_index(drop=True)
        print(f"  {len(df)} passages SNCF")
    except requests.RequestException as e:
        print(f"  ⚠  SNCF inaccessible ({e}) → fallback OSM")

    if df.empty:
        print("  → Fallback OSM...")
        df = _pn_from_osm()

    if df.empty or not pts:
        return df

    tree = build_kdtree(pts)
    df["dist_ligne_km"] = dist_to_line_km(
        df["lat"].to_numpy(), df["lon"].to_numpy(), tree
    )
    n_before = len(df)
    df = df[df["dist_ligne_km"] <= RAYON_KM].reset_index(drop=True)
    print(f"  → {len(df)}/{n_before} passages retenus (≤ {RAYON_KM} km de la ligne)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tunnels (embarqués)
# ─────────────────────────────────────────────────────────────────────────────

def get_tunnels() -> pd.DataFrame:
    """Retourne le DataFrame des 12 tunnels vérifiés de la ligne."""
    df = pd.DataFrame(TUNNELS)
    print(f"  {len(df)} tunnels chargés :")
    for _, t in df.iterrows():
        print(f"    {t['nom']:35s}  L={int(t['longueur_m'])} m  "
              f"entrée ({t['lat_entree']:.4f}, {t['lon_entree']:.4f})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Antennes ANFR
# ─────────────────────────────────────────────────────────────────────────────

def _parse_coordonnees(coord_str: str | None) -> tuple[float | None, float | None]:
    if not coord_str:
        return None, None
    try:
        parts = str(coord_str).split(",")
        if len(parts) == 2:
            return float(parts[0].strip()), float(parts[1].strip())
    except (ValueError, AttributeError):
        pass
    return None, None


def _extract_bande_mhz(systeme: str | None) -> int | None:
    if not systeme:
        return None
    for token in str(systeme).split():
        try:
            val = int(token)
            if 100 <= val <= 40_000:
                return val
        except ValueError:
            continue
    return None


_STATUTS_VALIDES = {"En service", "Techniquement opérationnel", "Approuvé"}


def fetch_antennes(pts: list[dict]) -> pd.DataFrame:
    """Télécharge les antennes ANFR (2G-5G) et filtre celles à moins de RAYON_KM du tracé."""
    all_records: list[dict] = []
    for dept in ANFR_DEPTS:
        offset, limit = 0, 100
        while True:
            params = {
                "dataset": "observatoire_2g_3g_4g",
                "refine.sta_nm_dpt": dept,
                "rows": limit,
                "start": offset,
            }
            r = requests.get(ANFR_URL, params=params, timeout=60)
            r.raise_for_status()
            data    = r.json()
            records = data.get("records", [])
            if not records:
                break
            all_records.extend(records)
            total = data.get("nhits", 0)
            print(f"  ANFR dept {dept} : {min(offset + limit, total)}/{total}...", end="\r")
            offset += limit
            if offset >= total:
                break
            time.sleep(0.1)
        print(f"  ANFR dept {dept} : {sum(1 for r in all_records if r.get('fields', {}).get('sta_nm_dpt') == dept or True)} records chargés")

    seen: set[str] = set()
    unique = [r for r in all_records if not (r.get("recordid", "") in seen or seen.add(r.get("recordid", "")))]  # type: ignore[arg-type]
    print(f"  Total dédoublonné : {len(unique)} enregistrements")

    rows = []
    for rec in unique:
        f = rec.get("fields", {})
        if f.get("statut") not in _STATUTS_VALIDES:
            continue
        lat, lon = _parse_coordonnees(f.get("coordonnees"))
        if lat is None or lon is None:
            continue
        rows.append({
            "ant_id":     str(rec.get("recordid", "")),
            "support_id": f.get("sup_id", ""),
            "operateur":  f.get("adm_lb_nom", ""),
            "generation": f.get("generation", ""),
            "bande_mhz":  _extract_bande_mhz(f.get("emr_lb_systeme")),
            "systeme":    f.get("emr_lb_systeme", ""),
            "statut":     f.get("statut", ""),
            "ant_lat":    lat,
            "ant_lon":    lon,
            "commune":    f.get("adr_lb_lieu", ""),
            "code_insee": f.get("code_insee", ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  ⚠  Aucune antenne valide récupérée.")
        return df

    print(f"  Antennes valides (statut OK + coordonnées) : {len(df)}")

    tree = build_kdtree(pts)
    df["dist_ligne_km"] = dist_to_line_km(
        df["ant_lat"].to_numpy(), df["ant_lon"].to_numpy(), tree
    )
    df = df[df["dist_ligne_km"] <= RAYON_KM].reset_index(drop=True)
    gen_counts = df["generation"].value_counts().to_dict()
    print(f"  → {len(df)} antennes dans {RAYON_KM} km de la ligne : {gen_counts}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Altitudes EU-DEM 25 m — points du tracé uniquement
# ─────────────────────────────────────────────────────────────────────────────

def fetch_elevations(pts: list[dict]) -> dict[int, float | None]:
    """Interroge OpenTopoData EU-DEM 25m en batches pour les points du tracé."""
    batches = [pts[i : i + TOPO_BATCH] for i in range(0, len(pts), TOPO_BATCH)]
    elevs: dict[int, float | None] = {}

    for i, batch in enumerate(batches):
        print(f"  EU-DEM batch {i + 1}/{len(batches)}...", end="\r")
        locations_str = "|".join(f"{p['lat']},{p['lon']}" for p in batch)
        try:
            r = requests.get(
                TOPO_URL,
                params={"locations": locations_str, "interpolation": "bilinear"},
                timeout=30,
            )
            r.raise_for_status()
            data    = r.json()
            status  = data.get("status", "")
            results = data.get("results", [])

            if status != "OK":
                raise ValueError(f"Statut API inattendu : '{status}'")
            if len(results) != len(batch):
                raise ValueError(
                    f"Réponse incomplète : {len(results)} résultats pour {len(batch)} points."
                )
            for j, res in enumerate(results):
                elevs[batch[j]["point_id"]] = res.get("elevation")

        except (requests.RequestException, ValueError) as e:
            print(f"\n  ⚠  Batch {i + 1} échoué ({e}) — altitudes mises à None.")
            for p in batch:
                elevs[p["point_id"]] = None

        if i < len(batches) - 1:
            time.sleep(1.1)  # rate-limit : 1 req/s

    print()
    n_ok = sum(1 for v in elevs.values() if v is not None)
    print(f"  {n_ok}/{len(pts)} altitudes récupérées")
    return elevs


# ─────────────────────────────────────────────────────────────────────────────
# 6. Raster DEM local pour le test LOS de prepare_data.py
# ─────────────────────────────────────────────────────────────────────────────

def _grid_bbox(pts: list[dict], rayon_km: float, margin_km: float) -> tuple[float, float, float, float, float]:
    """Calcule la bbox (lat_min, lat_max, lon_min, lon_max) + latitude moyenne."""
    lats = [p["lat"] for p in pts]
    lons = [p["lon"] for p in pts]
    lat_mean = sum(lats) / len(lats)

    buf_km = rayon_km + margin_km
    buf_deg_lat = buf_km / 111.0
    buf_deg_lon = buf_km / (111.0 * math.cos(math.radians(lat_mean)))

    lat_min = min(lats) - buf_deg_lat
    lat_max = max(lats) + buf_deg_lat
    lon_min = min(lons) - buf_deg_lon
    lon_max = max(lons) + buf_deg_lon
    return lat_min, lat_max, lon_min, lon_max, lat_mean


def _query_dem_batch(batch_coords: list[tuple[float, float]]) -> list[float | None]:
    """
    Une requête OpenTopoData, avec retries. Retourne une liste d'altitudes
    (ou None par point si tous les essais échouent). Ne jamais laisser un
    batch NaN silencieusement après un seul échec réseau ponctuel.
    """
    locations_str = "|".join(f"{la},{lo}" for la, lo in batch_coords)
    last_error = None
    for attempt in range(1, DEM_BATCH_MAX_RETRIES + 1):
        try:
            r = requests.get(
                TOPO_URL,
                params={"locations": locations_str, "interpolation": "bilinear"},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "OK":
                raise ValueError(f"statut API : {data.get('status')}")
            results = data.get("results", [])
            if len(results) != len(batch_coords):
                raise ValueError(f"{len(results)} résultats pour {len(batch_coords)} points")
            return [res.get("elevation") for res in results]
        except Exception as e:
            last_error = e
            if attempt < DEM_BATCH_MAX_RETRIES:
                time.sleep(DEM_BATCH_RETRY_DELAY_S)
    print(f"\n  ⚠ Batch DEM abandonné après {DEM_BATCH_MAX_RETRIES} essais ({last_error})")
    return [None] * len(batch_coords)


def build_local_dem_raster(
    pts: list[dict],
    rayon_km: float = RAYON_KM,
    margin_km: float = DEM_BUFFER_MARGIN_KM,
    grid_spacing_m: int = DEM_GRID_SPACING_M,
) -> Path | None:
    """
    Construit un raster DEM local (GeoTIFF, EPSG:4326) en interrogeant
    OpenTopoData une seule fois sur une grille régulière couvrant tout le
    corridor d'étude. C'est ce raster que prepare_data.py._init_dem() lit
    pour le test LOS, à la place de dizaines de milliers d'appels API
    individuels pendant le prétraitement.

    Si trop de points restent introuvables après retries (> DEM_MAX_NAN_FRACTION),
    le raster est refusé plutôt qu'écrit à moitié interpolé : mieux vaut un
    échec visible qu'un DEM dégradé qui reproduit silencieusement le bug
    qu'il est censé corriger.
    """
    lat_min, lat_max, lon_min, lon_max, lat_mean = _grid_bbox(pts, rayon_km, margin_km)

    dlat = grid_spacing_m / 111_000.0
    dlon = grid_spacing_m / (111_000.0 * math.cos(math.radians(lat_mean)))

    n_rows = int((lat_max - lat_min) / dlat) + 1
    n_cols = int((lon_max - lon_min) / dlon) + 1
    n_pts  = n_rows * n_cols

    print(f"  Zone couverte : lat [{lat_min:.4f}, {lat_max:.4f}] "
          f"lon [{lon_min:.4f}, {lon_max:.4f}]")
    print(f"  Grille DEM : {n_rows} x {n_cols} = {n_pts} points (pas {grid_spacing_m} m)")

    grid_lats = [lat_max - i * dlat for i in range(n_rows)]
    grid_lons = [lon_min + j * dlon for j in range(n_cols)]
    query_points = [(la, lo) for la in grid_lats for lo in grid_lons]

    elevations = np.full(n_pts, np.nan, dtype="float64")
    n_batches = math.ceil(n_pts / TOPO_BATCH)

    for i in range(n_batches):
        batch = query_points[i * TOPO_BATCH : (i + 1) * TOPO_BATCH]
        print(f"  DEM raster batch {i + 1}/{n_batches}...", end="\r")
        values = _query_dem_batch(batch)
        for j, v in enumerate(values):
            elevations[i * TOPO_BATCH + j] = v if v is not None else np.nan
        if i < n_batches - 1:
            time.sleep(1.1)  # rate-limit : 1 req/s

    print()
    grid = elevations.reshape(n_rows, n_cols)

    n_nan = int(np.isnan(grid).sum())
    nan_fraction = n_nan / n_pts
    if n_nan:
        print(f"  {n_nan}/{n_pts} points manquants ({nan_fraction:.1%}) même après retries.")

    if nan_fraction > DEM_MAX_NAN_FRACTION:
        print(f"  ✗ Raster REFUSÉ : {nan_fraction:.1%} de points manquants "
              f"(seuil toléré {DEM_MAX_NAN_FRACTION:.0%}).")
        print("    Causes probables : rate-limit OpenTopoData atteint, ou coupure")
        print("    réseau pendant les ~15-20 min de la collecte. Relancez cette étape")
        print("    seule (le script réinterroge tout, il n'y a pas de reprise partielle).")
        return None

    if n_nan:
        from scipy.interpolate import griddata
        yy, xx = np.mgrid[0:n_rows, 0:n_cols]
        valid = ~np.isnan(grid)
        grid[~valid] = griddata(
            (yy[valid], xx[valid]), grid[valid],
            (yy[~valid], xx[~valid]), method="nearest",
        )

    TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TERRAIN_DIR / "eu_dem_courpiere_ambert.tif"

    transform = from_origin(lon_min, lat_max, dlon, dlat)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=n_rows, width=n_cols,
        count=1, dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(grid.astype("float32"), 1)

    # Vérification systématique post-écriture : un fichier de 0 octet ou
    # illisible est le signe le plus courant d'un raster "qui ne s'enregistre
    # pas" alors même que le code semblait s'être exécuté jusqu'au bout.
    if not out_path.exists() or out_path.stat().st_size == 0:
        print(f"  ✗ Écriture échouée : {out_path} absent ou vide.")
        return None
    try:
        with rasterio.open(out_path) as check:
            check.read(1, window=((0, 1), (0, 1)))
    except Exception as e:
        print(f"  ✗ Fichier écrit mais illisible par rasterio ({e}).")
        return None

    print(f"  ✓ Raster DEM local sauvegardé : {out_path} ({out_path.stat().st_size/1024:.0f} Ko)")
    print(f"    ({n_rows}x{n_cols} px, pas {grid_spacing_m}m, "
          f"alt [{np.nanmin(grid):.0f}–{np.nanmax(grid):.0f} m])")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("ÉTAPE 1 — fetch_data.py")
    print("Sources : CSV local · SNCF · ANFR · OpenTopoData EU-DEM")
    print("Tracé   : data/raw/courpiere_ambert_resampled.csv")
    print("Tunnels : 12 tunnels embarqués (Courpière–Ambert, vérifiés)")
    print("=" * 70)

    print("\n[1/6] Tracé ferroviaire (lecture CSV)...")
    trace_csv = RAW_DIR / "courpiere_ambert_resampled.csv"
    try:
        pts = load_trace(trace_csv, spacing_m=SPACING_M)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ Erreur critique : {e}")
        sys.exit(1)

    print("\n[2/6] Passages à niveau (SNCF + fallback OSM)...")
    df_pn = fetch_passages_niveau(pts)

    print("\n[3/6] Tunnels...")
    df_tunnels = get_tunnels()

    print("\n[4/6] Antennes mobiles (ANFR)...")
    df_ant = fetch_antennes(pts)

    n_batches = math.ceil(len(pts) / TOPO_BATCH)
    print(f"\n[5/6] Altitudes EU-DEM du tracé ({n_batches} requête(s))...")
    elevs    = fetch_elevations(pts)
    df_ligne = pd.DataFrame(pts)
    df_ligne["altitude_m"] = df_ligne["point_id"].map(elevs)
    df_ligne["altitude_m"] = (
        pd.to_numeric(df_ligne["altitude_m"], errors="coerce")
          .interpolate("linear")
          .round(1)
    )

    print(f"\n[6/6] Raster DEM local (zone complète, pour le test LOS)...")
    dem_path = build_local_dem_raster(pts, rayon_km=RAYON_KM, margin_km=DEM_BUFFER_MARGIN_KM,
                                      grid_spacing_m=DEM_GRID_SPACING_M)

    df_ligne.to_csv(RAW_DIR / "ligne_gps.csv",       index=False)
    df_pn.to_csv(RAW_DIR   / "passages_niveau.csv",  index=False)
    df_tunnels.to_csv(RAW_DIR / "tunnels.csv",        index=False)
    df_ant.to_csv(RAW_DIR  / "antennes_anfr.csv",     index=False)

    print("\n" + "─" * 70)
    print(f"✓ Fichiers sauvegardés dans {RAW_DIR}/")
    alt_min = df_ligne["altitude_m"].min()
    alt_max = df_ligne["altitude_m"].max()
    print(f"  ligne_gps.csv       : {len(df_ligne):>5} points "
          f"[{df_ligne['distance_km'].max():.1f} km, alt {alt_min:.0f}–{alt_max:.0f} m]")
    print(f"  passages_niveau.csv : {len(df_pn):>5} passages")
    print(f"  tunnels.csv         : {len(df_tunnels):>5} tunnels")
    if not df_ant.empty:
        gen = df_ant["generation"].value_counts().to_dict()
        print(f"  antennes_anfr.csv   : {len(df_ant):>5} antennes {gen}")
    else:
        print(f"  antennes_anfr.csv   :     0 antennes")

    print()
    if dem_path is not None:
        print(f"  ✓✓✓ terrain/eu_dem_courpiere_ambert.tif : PRÊT pour prepare_data.py")
    else:
        print(f"  ✗✗✗ RASTER DEM NON GÉNÉRÉ — prepare_data.py va retomber sur l'API en")
        print(f"      direct pour le test LOS (lent, risque de fallback dégradé sur la")
        print(f"      voie ferrée). Voir le message d'erreur ci-dessus pour la cause")
        print(f"      exacte et relancez l'étape 6/6.")

    print("\n→ Prochaine étape : python fetch_temperature.py")
    print("  puis              : python prepare_data.py")


if __name__ == "__main__":
    main()