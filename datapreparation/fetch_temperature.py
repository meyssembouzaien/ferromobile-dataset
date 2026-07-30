"""
fetch_temperature.py — Collecte des données météo horaires ERA5
===============================================================
Ce script récupère les conditions météorologiques horaires réelles
depuis l'API Open-Meteo (réanalyse ERA5) pour plusieurs scénarios
saisonniers représentatifs de la ligne Courpière–Ambert.

Pourquoi des données horaires ?
  Une moyenne annuelle ou mensuelle efface la variabilité temporelle
  (différence jour/nuit, événements de pluie, etc.) qui est pourtant
  essentielle pour entraîner un agent DQN robuste à des conditions
  variées. En fetchant des journées réelles complètes (24 h), on obtient
  une météo physiquement cohérente sans aucun modèle synthétique.

Approche adoptée — scénarios saisonniers :
  On sélectionne 4 journées représentatives de 2023, une par saison,
  choisies parmi les journées météo les plus typiques de la région
  (données ERA5). Pour chaque journée, on dispose de 24 valeurs horaires
  par variable (température, pluie, vent, humidité, nébulosité).

  Ces scénarios sont ensuite interpolés spatialement le long de la ligne
  (par cluster de 10 km) pour produire un DataFrame avec les dimensions :
    point_id × scenario_id × heure → variables météo

  Sur 30 km avec CLUSTER_KM=10 → 4 clusters × 4 scénarios × 24 h
  = ~300 points × 96 situations météo ≈ 28 800 lignes (taille raisonnable)

Scénarios utilisés (modifiables dans SCENARIOS) :
  S1 — Hiver   : 2023-01-15  (froid, peu de pluie, vent modéré)
  S2 — Printemps : 2023-04-10 (doux, pluie probable, vent faible)
  S3 — Été     : 2023-07-20  (chaud, pluie faible, rayonnement fort)
  S4 — Automne : 2023-10-12  (frais, pluie fréquente, vent fort)

Sortie :
  data/raw/temperature_scenarios.csv
  Colonnes : point_id, lat, lon, distance_km, altitude_m,
             scenario_id, saison, date, heure,
             temperature_c, precipitation_mm_h, windspeed_kmh,
             humidity_pct, cloudcover_pct

Usage :
  pip install requests pandas numpy
  python fetch_temperature.py
"""

import json
import math
import os
import time

import numpy as np
import pandas as pd
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Chemins relatifs au dossier du script
# data/ est à la racine du projet, scripts dans datapreparation/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)  # remonte d'un niveau → ferromobile_project/
INPUT_LIGNE  = os.path.join(BASE_DIR, "data", "raw", "ligne_gps.csv")
OUTPUT_FILE  = os.path.join(BASE_DIR, "data", "raw", "temperature_scenarios.csv")
CACHE_FILE   = os.path.join(BASE_DIR, "data", "raw", "temperature_cache.json")

os.makedirs(os.path.join(BASE_DIR, "data", "raw"), exist_ok=True)

# Tronçon étudié : Courpière–Ambert (30 km)
# ligne_gps.csv contient toute la ligne 785000 (634 km) —
# on filtre ici pour ne garder que le tronçon d'intérêt.
TRONCON_KM_MIN = 0.0
TRONCON_KM_MAX = 30.0

# Résolution spatiale des clusters (km).
# Sur 30 km avec CLUSTER_KM=10 → 4 clusters, ce qui est cohérent
# avec la résolution native ERA5 (~31 km) et suffisant pour
# capturer le gradient spatial sur ce tronçon court.
CLUSTER_KM = 10.0

# Délai entre requêtes API (secondes) — Open-Meteo limite à ~30 req/min
API_DELAY_S = 2.5

# Note : si vous changez les SCENARIOS, supprimez le cache pour forcer le re-fetch :
#   rm data/raw/temperature_cache.json

# Scénarios saisonniers : (identifiant, saison, date au format YYYY-MM-DD)
# Dates sélectionnées pour représenter des conditions météo variées,
# notamment avec des précipitations significatives, sur la région Ambert/Livradois.
# Méthode de sélection : journées représentatives de chaque saison avec
# précipitations ≥ 2 mm/jour selon les archives climatiques ERA5 2023.
#
# S1 — Hiver     : pluie hivernale persistante + froid
# S2 — Printemps : pluie de printemps modérée + douceur
# S3 — Été       : orage convectif intense (pic d'atténuation radio)
# S4 — Automne   : épisode pluvieux automnal + vent fort
SCENARIOS = [
    ("S1", "hiver",     "2023-12-10"),   # était 2023-01-18 — pluie hivernale 4.9mm, pic 1mm/h
    ("S2", "printemps", "2023-04-19"),   # était 2023-04-02 — pic printanier 1.7mm/h
    ("S3", "ete",       "2023-06-30"),   # conservé — orage convectif 8.7mm/h ✅
    ("S4", "automne",   "2023-10-20"),   # conservé — pluie + vent 29.5km/h ✅
]

# Heure de référence par scénario utilisée dans enrich_data.py pour l'extraction
# de la valeur météo horaire ERA5 (pic de précipitations de la journée).
# Ces heures correspondent aux pics de pluie observés sur chaque journée ERA5.
HEURE_PAR_SCENARIO = {
    "S1": 19,  # 0.80 mm/h — pic soirée hivernale
    "S2": 19,  # 1.60 mm/h — orage printanier soirée
    "S3":  7,  # 6.87 mm/h — orage convectif matinal
    "S4":  5,  # 3.90 mm/h — front automnal nuit/matin
}
# URL de l'API Open-Meteo (réanalyse ERA5)
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points GPS (formule de Haversine)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def load_cache():
    """
    Charge le cache local des requêtes ERA5 déjà effectuées.

    Le cache évite de retélécharger les données si le script est
    relancé après une interruption (rate limit, coupure réseau, etc.).

    Retourne : dict {clé_cache → données horaires}
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
            print(f"  ✓ Cache chargé : {len(data)} entrées")
            return data
        except Exception as e:
            print(f"  ⚠ Cache illisible ({e}) — reparti de zéro")
    return {}


def save_cache(cache):
    """Sauvegarde le cache sur disque après chaque requête réussie."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def cache_key(lat, lon, date):
    """Clé unique pour identifier une requête (point × date)."""
    return f"{lat:.4f},{lon:.4f},{date}"


# ──────────────────────────────────────────────────────────────────────────────
# Fetch ERA5 horaire pour un point et une date
# ──────────────────────────────────────────────────────────────────────────────

def fetch_era5_horaire(lat, lon, date, cache):
    """
    Récupère les 24 heures de données météo ERA5 pour un point GPS et une date.

    Utilise le cache local pour éviter les requêtes répétées.
    Applique un backoff exponentiel en cas de rate limit (HTTP 429).

    Paramètres :
      lat, lon : coordonnées GPS du cluster (degrés décimaux)
      date     : date au format 'YYYY-MM-DD'
      cache    : dict de cache modifié en place

    Retourne :
      DataFrame avec 24 lignes (une par heure) et les colonnes météo,
      ou None si la requête échoue.
    """
    key = cache_key(lat, lon, date)
    if key in cache:
        return pd.DataFrame(cache[key])

    params = {
        "latitude":   round(lat, 4),
        "longitude":  round(lon, 4),
        "start_date": date,
        "end_date":   date,
        "hourly": (
            "temperature_2m,precipitation,windspeed_10m,"
            "relativehumidity_2m,cloudcover"
        ),
        "timezone":   "Europe/Paris",
        "models":     "era5",
        "wind_speed_unit": "kmh",
    }

    # Backoff exponentiel : 30 s → 60 s → 120 s en cas de rate limit
    backoff = [30, 60, 120]
    for attempt in range(len(backoff) + 1):
        try:
            r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
            if r.status_code == 429 and attempt < len(backoff):
                wait = backoff[attempt]
                print(f"\n  ⏳ Rate limit — attente {wait}s "
                      f"(tentative {attempt + 1}/{len(backoff) + 1})...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == len(backoff):
                print(f"\n  ✗ Échec définitif pour ({lat:.4f}, {lon:.4f}, {date}) : {e}")
                return None
            time.sleep(backoff[attempt])

    h = r.json().get("hourly", {})
    times = h.get("time", [])
    if not times:
        return None

    rows = []
    for i, t in enumerate(times):
        heure = int(t.split("T")[1].split(":")[0]) if "T" in t else i
        rows.append({
            "heure":             heure,
            "temperature_c":     h["temperature_2m"][i],
            "precipitation_mm_h": h["precipitation"][i],
            "windspeed_kmh":     h["windspeed_10m"][i],
            "humidity_pct":      h["relativehumidity_2m"][i],
            "cloudcover_pct":    h["cloudcover"][i],
        })

    df_h = pd.DataFrame(rows)

    # Remplacer les NaN par interpolation linéaire
    for col in df_h.columns:
        if col != "heure":
            df_h[col] = df_h[col].interpolate("linear").ffill().bfill()

    # Sauvegarder dans le cache
    cache[key] = df_h.to_dict(orient="records")
    save_cache(cache)

    return df_h


# ──────────────────────────────────────────────────────────────────────────────
# Construction des clusters spatiaux
# ──────────────────────────────────────────────────────────────────────────────

def build_clusters(df_ligne):
    """
    Groupe les points GPS en clusters spatiaux espacés de CLUSTER_KM km.

    Un cluster représente une zone géographique homogène pour ERA5.
    Les données météo sont fetchées une fois par cluster, puis interpolées
    linéairement entre clusters pour couvrir tous les points GPS.

    Paramètre  : df_ligne — DataFrame avec colonnes lat, lon, distance_km, altitude_m
    Retourne   : liste de dicts {cluster_id, lat, lon, distance_km, altitude_m}
    """
    clusters = []
    last_dist = -CLUSTER_KM

    for _, pt in df_ligne.iterrows():
        if pt["distance_km"] - last_dist >= CLUSTER_KM:
            clusters.append({
                "cluster_id":  len(clusters),
                "lat":         round(float(pt["lat"]), 4),
                "lon":         round(float(pt["lon"]), 4),
                "distance_km": float(pt["distance_km"]),
                "altitude_m":  float(pt.get("altitude_m", 400.0)),
            })
            last_dist = pt["distance_km"]

    return clusters


# ──────────────────────────────────────────────────────────────────────────────
# Correction altitudinale de la température
# ──────────────────────────────────────────────────────────────────────────────

def apply_altitude_correction(temp_cluster_c, alt_cluster_m, alt_point_m):
    """
    Corrige la température d'un point GPS par rapport à l'altitude du cluster ERA5.

    Loi utilisée : gradient adiabatique standard de l'atmosphère = −6.5°C / 1000 m
    (OMM / ICAO Standard Atmosphere, 1975).

    Paramètres :
      temp_cluster_c : température au niveau du cluster ERA5 (°C)
      alt_cluster_m  : altitude du cluster ERA5 (m)
      alt_point_m    : altitude du point GPS cible (m)

    Retourne : température corrigée (°C), arrondie à 2 décimales.
    """
    correction = -0.0065 * (alt_point_m - alt_cluster_m)
    return round(temp_cluster_c + correction, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("fetch_temperature.py — Météo horaire ERA5 par scénario saisonnier")
    print(f"  {len(SCENARIOS)} scénarios × clusters {CLUSTER_KM} km")
    print("=" * 70)

    # Chargement de la ligne GPS
    print(f"\n[1/3] Chargement de {INPUT_LIGNE}...")
    if not os.path.exists(INPUT_LIGNE):
        raise FileNotFoundError(
            f"{INPUT_LIGNE} introuvable — lancez d'abord fetch_data.py"
        )
    df_ligne = pd.read_csv(INPUT_LIGNE)
    print(f"  {len(df_ligne)} points GPS chargés (ligne complète 785000)")

    # Filtrage sur le tronçon Courpière–Ambert uniquement.
    # ligne_gps.csv contient toute la ligne 785000 (634 km) ;
    # on recentre la distance sur 0 puis on filtre sur 30 km.
    df_ligne["distance_km"] = (
        df_ligne["distance_km"] - df_ligne["distance_km"].min()
    ).round(4)
    df_ligne = df_ligne[
        (df_ligne["distance_km"] >= TRONCON_KM_MIN) &
        (df_ligne["distance_km"] <= TRONCON_KM_MAX)
    ].copy().reset_index(drop=True)
    print(f"  → {len(df_ligne)} points après filtrage tronçon "
          f"[{TRONCON_KM_MIN}–{TRONCON_KM_MAX} km] | "
          f"alt {df_ligne['altitude_m'].min():.0f}–{df_ligne['altitude_m'].max():.0f} m")

    # Construction des clusters spatiaux
    clusters = build_clusters(df_ligne)
    print(f"  {len(clusters)} clusters ERA5 (résolution {CLUSTER_KM} km)")
    for cl in clusters:
        print(f"    Cluster {cl['cluster_id']} : "
              f"dist={cl['distance_km']:.1f} km, alt={cl['altitude_m']:.0f} m, "
              f"({cl['lat']:.4f}, {cl['lon']:.4f})")

    # Fetch ERA5 pour chaque (cluster × scénario)
    print(f"\n[2/3] Fetch ERA5 — {len(SCENARIOS)} scénarios × {len(clusters)} clusters "
          f"= {len(SCENARIOS) * len(clusters)} requêtes maximum...")
    cache = load_cache()
    all_rows = []
    n_total  = len(SCENARIOS) * len(clusters)
    n_done   = 0

    for scenario_id, saison, date in SCENARIOS:
        print(f"\n  Scénario {scenario_id} — {saison} ({date})")

        # Fetch pour chaque cluster
        cluster_data = {}   # cluster_id → DataFrame horaire (24 h)
        for cl in clusters:
            n_done += 1
            print(f"    Cluster {cl['cluster_id']+1}/{len(clusters)} "
                  f"({cl['lat']:.4f}, {cl['lon']:.4f})... [{n_done}/{n_total}]",
                  end="\r", flush=True)

            df_h = fetch_era5_horaire(cl["lat"], cl["lon"], date, cache)
            if df_h is not None:
                cluster_data[cl["cluster_id"]] = df_h
            time.sleep(API_DELAY_S)

        if not cluster_data:
            print(f"\n  ✗ Aucune donnée pour {scenario_id} — scénario ignoré")
            continue

        # Interpolation spatiale : pour chaque point GPS, on interpole
        # linéairement entre les clusters selon la distance cumulée
        cl_dists = np.array([cl["distance_km"] for cl in clusters])
        pt_dists = df_ligne["distance_km"].values
        pt_alts  = df_ligne["altitude_m"].values

        meteo_cols = [
            "temperature_c", "precipitation_mm_h",
            "windspeed_kmh", "humidity_pct", "cloudcover_pct"
        ]

        # Pour chaque heure (0–23), construire une ligne par point GPS
        n_heures = 24
        for heure in range(n_heures):
            # Valeurs des clusters à cette heure
            cl_vals = {}
            for col in meteo_cols:
                cl_vals[col] = []
                for cl in clusters:
                    df_cl = cluster_data.get(cl["cluster_id"])
                    if df_cl is not None and heure < len(df_cl):
                        cl_vals[col].append(float(df_cl.iloc[heure][col]
                                            if col in df_cl.columns else 0.0))
                    else:
                        cl_vals[col].append(0.0)

            # Interpolation spatiale par np.interp
            interp = {}
            for col in meteo_cols:
                vals = np.array(cl_vals[col])
                interp[col] = np.interp(pt_dists, cl_dists, vals)

            # Correction altitudinale de la température
            # Altitude de référence du cluster interpolé au point courant
            cl_alts = np.array([cl["altitude_m"] for cl in clusters])
            alt_cluster_interp = np.interp(pt_dists, cl_dists, cl_alts)

            for i, (_, pt) in enumerate(df_ligne.iterrows()):
                temp_corr = apply_altitude_correction(
                    interp["temperature_c"][i],
                    alt_cluster_interp[i],
                    pt_alts[i]
                )

                all_rows.append({
                    "point_id":           int(pt["point_id"]),
                    "lat":                round(float(pt["lat"]), 6),
                    "lon":                round(float(pt["lon"]), 6),
                    "distance_km":        round(float(pt["distance_km"]), 4),
                    "altitude_m":         round(float(pt_alts[i]), 1),
                    "scenario_id":        scenario_id,
                    "saison":             saison,
                    "date":               date,
                    "heure":              heure,
                    "temperature_c":      temp_corr,
                    "precipitation_mm_h": round(float(interp["precipitation_mm_h"][i]), 4),
                    "windspeed_kmh":      round(float(interp["windspeed_kmh"][i]), 2),
                    "humidity_pct":       round(float(interp["humidity_pct"][i]), 1),
                    "cloudcover_pct":     round(float(interp["cloudcover_pct"][i]), 1),
                })

        n_pts_ok = len(df_ligne)
        print(f"\n  ✓ {scenario_id} — {n_pts_ok} points × 24 h = "
              f"{n_pts_ok * 24} lignes générées")

    # Sauvegarde
    print(f"\n[3/3] Sauvegarde...")
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUTPUT_FILE, index=False)

    # Résumé
    print(f"\n✓ {OUTPUT_FILE}")
    print(f"  {len(df_out):,} lignes × {len(df_out.columns)} colonnes")
    print(f"  Scénarios  : {df_out['scenario_id'].nunique()} "
          f"({df_out['scenario_id'].unique().tolist()})")
    print(f"  Points GPS : {df_out['point_id'].nunique()}")
    print(f"  Heures     : {df_out['heure'].nunique()} (0–23)")

    print("\n  Résumé par scénario :")
    print(f"  {'Scénario':<20} {'Temp min':>8} {'Temp moy':>8} {'Temp max':>8} "
          f"{'Pluie moy':>10} {'Pluie max':>10} {'Vent moy':>9}")
    print("  " + "-" * 80)
    for sc_id, saison, date in SCENARIOS:
        sc = df_out[df_out["scenario_id"] == sc_id]
        if sc.empty:
            continue
        t = sc["temperature_c"]
        p = sc["precipitation_mm_h"]
        v = sc["windspeed_kmh"]
        label = f"{sc_id} {saison} ({date})"
        print(f"  {label:<20} {t.min():>7.1f}°C {t.mean():>7.1f}°C {t.max():>7.1f}°C "
              f"  {p.mean():>8.3f}  {p.max():>9.3f}  {v.mean():>7.1f}")
        # Avertissement si pas de pluie
        if p.max() == 0:
            print(f"  ⚠ {sc_id} : pluie = 0 sur toute la journée — date sans précipitations")
        else:
            print(f"  ✓ {sc_id} : pluie présente (max={p.max():.3f} mm/h)")

    print("\n→ Prochaine étape : python prepare_data.py")
    print("  puis              : python enrich_data.py")


if __name__ == "__main__":
    main()