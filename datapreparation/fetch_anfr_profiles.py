"""
fetch_anfr_profiles.py — Profils technologiques depuis les antennes ANFR réelles
=================================================================================
Source : data/raw/antennes_anfr.csv  (antennes réelles zone Courpière-Ambert)
         data/raw/ligne_gps.csv      (tracé réel avec altitudes EU-DEM)

Ce script remplace les valeurs génériques de la littérature par des valeurs
calculées depuis les vraies antennes ANFR présentes dans la zone.

Méthode :
  Pour chaque (technologie, bande) présent dans antennes_anfr.csv :
  1. Calculer la distance médiane train-antenne (depuis dist_ligne_km)
  2. Récupérer l'altitude médiane des antennes (depuis EU-DEM cache ou API)
  3. Appliquer le modèle de propagation adapté :
       - COST 231-Hata pour f ≤ 2000 MHz (2G/3G/4G basses bandes)
       - 3GPP TR 38.901 RMa  pour f > 2000 MHz (4G hautes bandes, 5G)  ← FIX A2
  4. Calculer SNR réel avec P_tx correct par technologie               ← FIX A3
  5. Calculer débit Shannon avec BW réelle par (techno, bande)         ← FIX A4

CORRECTIONS v2 :
  A2 — COST 231-Hata hors plage f>2000 MHz → 3GPP TR 38.901 RMa
  A3 — P_tx unique 46 dBm → P_tx par technologie (43/43/46/49 dBm)
  A4 — BW_MHZ générique → BW réelle par (techno, bande) selon ARCEP

CORRECTIONS v3 :
  B1 — RTT_RANGE 2G : (650, 1000) → (300, 700) ms
       Ref : GSMA IR.92 §6.2 ; mesures terrain GSM-R ferroviaire (UIC 857).
  B2 — DEBIT_PLAFOND_REEL par (techno, bande) : remplace plafond_absolu Shannon(30dB)
       5G/3500 MHz : Shannon(30dB, 80MHz) ≈ 797 Mbps → plafonné à 300 Mbps (rural NR n78)
       5G/2100 MHz : plafonné à 250 Mbps (rural sub-6GHz)
       5G/700 MHz  : plafonné à 150 Mbps (bande 700 MHz, 10 MHz BW)
       Ref : 3GPP TR 38.913 §7.1 ; benchmarks ruraux Orange/SFR France 2023.

Sortie :
  data/raw/anfr_profiles.json
  → chargé automatiquement par prepare_data.py

Usage :
  python fetch_anfr_profiles.py
"""

import json
import math
import os
import time
import urllib.request

import pandas as pd
import numpy as np

INPUT_DIR   = "data/raw"
OUTPUT_DIR  = "data/raw"
CACHE_DIR   = "data/cache"
OUTPUT_FILE = f"{OUTPUT_DIR}/anfr_profiles.json"
ELEV_CACHE_FILE = f"{CACHE_DIR}/elev_cache.json"

os.makedirs(CACHE_DIR, exist_ok=True)

PORTEE_KM = {
    ("2G",   900): 20.0,
    ("3G",   900): 18.5,
    ("3G",  2100): 7.0,
    ("4G",   700): 15.0,
    ("4G",   800): 17.0,
    ("4G",  1800): 7.0,
    ("4G",  2100): 5.0,
    ("4G",  2600): 4.0,
    ("5G",   700): 10.0,
    ("5G",  2100): 5.0,
    ("5G",  3500): 2.0,
}

BER_BASE = {
    "2G": 1e-2,
    "3G": 1e-3,
    "4G": 5e-4,
    "5G": 1e-4,
}

# [FIX-B1] RTT 2G corrigé : (650, 1000) → (300, 700) ms
# Ref : GSMA IR.92 §6.2 ; mesures terrain GSM-R ferroviaire (UIC 857) → 300–600 ms.
# L'ancien plancher 650 ms donnait une médiane ~700 ms, trop élevée pour GPRS/EDGE.
RTT_RANGE = {
    "2G":  (300, 700),
    "3G":  (120, 160),
    "4G":  (30,  80),
    "5G":  (15,  35),
}

DELTA_SNR = {
    "2G": 3.0,
    "3G": 2.0,
    "4G": 0.5,
    "5G": 0.6,
}

IMPACT_VITESSE = {
    "2G": -0.21,
    "3G": -0.20,
    "4G": -0.05,
    "5G": -0.06,
}

MAT_HEIGHT_M = {
    "2G": 40.0,
    "3G": 38.0,
    "4G": 35.0,
    "5G": 25.0,
}

BW_MHZ_PAR_BANDE = {
    ("2G",   900): 0.2,
    ("3G",   900): 5.0,
    ("3G",  2100): 5.0,
    ("4G",   700): 10.0,
    ("4G",   800): 10.0,
    ("4G",  1800): 20.0,
    ("4G",  2100): 20.0,
    ("4G",  2600): 20.0,
    ("5G",   700): 10.0,
    ("5G",  2100): 20.0,
    ("5G",  3500): 80.0,
}

BW_MHZ_DEFAULT = {
    "2G": 0.2,
    "3G": 5.0,
    "4G": 10.0,
    "5G": 20.0,
}

# [FIX-B2] Plafonds réalistes de débit par (techno, bande) — contexte rural Courpière-Ambert
# Remplace le plafond absolu Shannon(SNR=30dB) qui produisait 797 Mbps pour 5G/3500 MHz.
# Ces plafonds correspondent aux débits crête mesurés en déploiement rural sub-6GHz France.
# Ref : 3GPP TR 38.913 §7.1 ; benchmarks Orange/SFR rural 2023 ; nPerf baromètre ARCEP.
DEBIT_PLAFOND_REEL = {
    ("2G",   900): 0.24,
    ("3G",   900): 7.2,
    ("3G",  2100): 7.2,
    ("4G",   700): 42.0,
    ("4G",   800): 42.0,
    ("4G",  1800): 75.0,
    ("4G",  2100): 60.0,
    ("4G",  2600): 90.0,
    ("5G",   700): 150.0,   # 10 MHz BW, rural → 100–150 Mbps réaliste
    ("5G",  2100): 250.0,   # 20 MHz BW, rural sub-6GHz → 200–250 Mbps
    ("5G",  3500): 300.0,   # 80 MHz BW, mais rural NR n78 → 300 Mbps (pas mmWave)
}

P_TX_DBM = {
    "2G": 43.0,
    "3G": 43.0,
    "4G": 46.0,
    "5G": 49.0,
}

NOISE_FLOOR_DBM = -100.0


# ──────────────────────────────────────────────────────────────────────────────
# Cache altitudes EU-DEM 25m
# ──────────────────────────────────────────────────────────────────────────────
_ELEV_CACHE = {}

def load_elev_cache():
    if os.path.exists(ELEV_CACHE_FILE):
        try:
            with open(ELEV_CACHE_FILE) as f:
                raw = json.load(f)
            for k, v in raw.items():
                lat, lon = map(float, k.split(","))
                _ELEV_CACHE[(lat, lon)] = v
            print(f"  ✓ Cache altitudes chargé : {len(_ELEV_CACHE)} points")
        except Exception as e:
            print(f"  ⚠ Cache illisible : {e}")


def fetch_elevations_batch(coords):
    BATCH = 100
    results = {}
    batches = [coords[i:i+BATCH] for i in range(0, len(coords), BATCH)]
    for bi, batch in enumerate(batches):
        print(f"  Batch {bi+1}/{len(batches)}...", end="\r", flush=True)
        locations = "|".join(f"{lat:.4f},{lon:.4f}" for lat, lon in batch)
        url = f"https://api.opentopodata.org/v1/eudem25m?locations={locations}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            for j, res in enumerate(data.get("results", [])):
                elev = res.get("elevation")
                if elev is not None:
                    results[batch[j]] = float(elev)
        except Exception as e:
            print(f"\n  ⚠ batch {bi+1} échoué : {e}")
        time.sleep(1.0)
    return results


def get_antenna_elevations(df_ant):
    load_elev_cache()
    coords_needed = []
    for _, row in df_ant.iterrows():
        key = (round(row["ant_lat"], 4), round(row["ant_lon"], 4))
        if key not in _ELEV_CACHE:
            coords_needed.append(key)
    if coords_needed:
        print(f"  {len(coords_needed)} altitudes antennes à fetcher...")
        fetched = fetch_elevations_batch(coords_needed)
        _ELEV_CACHE.update(fetched)
        print(f"\n  ✓ {len(fetched)} altitudes récupérées")
    else:
        print(f"  ✓ Toutes les altitudes antennes en cache")
    altitudes = []
    for _, row in df_ant.iterrows():
        key = (round(row["ant_lat"], 4), round(row["ant_lon"], 4))
        alt = _ELEV_CACHE.get(key, 700.0)
        altitudes.append(alt)
    return altitudes


# ──────────────────────────────────────────────────────────────────────────────
# Modèles de propagation
# ──────────────────────────────────────────────────────────────────────────────

def cost231_path_loss(dist_km, freq_mhz, h_bs_m, h_ms_m=4.0):
    freq_mhz = max(150, min(freq_mhz, 2000))
    d        = max(dist_km, 0.05)
    a_hms = ((1.1 * math.log10(freq_mhz) - 0.7) * h_ms_m
             - (1.56 * math.log10(freq_mhz) - 0.8))
    path_loss = (46.3
                 + 33.9 * math.log10(freq_mhz)
                 - 13.82 * math.log10(h_bs_m)
                 - a_hms
                 + (44.9 - 6.55 * math.log10(h_bs_m)) * math.log10(d)
                 - 4.78 * (math.log10(freq_mhz) ** 2)
                 + 18.33 * math.log10(freq_mhz)
                 - 35.94)
    return path_loss


def tr38901_path_loss(dist_km, freq_mhz, h_bs_m, h_ms_m=4.0):
    d_3d  = max(dist_km * 1000.0, 10.0)
    fc    = freq_mhz / 1000.0
    h_avg = 5.0
    pl1 = (20.0 * math.log10(40.0 * math.pi * d_3d * fc / 3.0)
           + min(0.03 * h_avg**1.72, 10.0) * math.log10(d_3d)
           - min(0.044 * h_avg**1.72, 14.77)
           + 0.002 * math.log10(h_avg) * d_3d)
    return pl1


def select_path_loss_model(dist_km, freq_mhz, h_bs_m, h_ms_m=4.0):
    if freq_mhz <= 2000:
        return cost231_path_loss(dist_km, freq_mhz, h_bs_m, h_ms_m)
    else:
        return tr38901_path_loss(dist_km, freq_mhz, h_bs_m, h_ms_m)


def shannon_mbps(snr_db, bw_mhz, debit_plafond):
    """Débit Shannon plafonné au plafond réaliste de la techno."""
    snr_lin = max(0.001, 10 ** (snr_db / 10))
    return round(min(bw_mhz * math.log2(1 + snr_lin), debit_plafond), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Calcul des profils depuis ANFR
# ──────────────────────────────────────────────────────────────────────────────

def compute_anfr_profiles(df_ant, df_ligne):
    print("\n[3/4] Calcul profils depuis antennes ANFR réelles...")

    alt_train_median = df_ligne["altitude_m"].median()
    print(f"  Altitude médiane train : {alt_train_median:.0f} m")

    profiles = {}
    stats    = []

    for (gen, bande), grp in df_ant.groupby(["generation", "bande_mhz"]):
        n_antennes = len(grp)
        freq_mhz   = int(bande)
        techno     = gen

        dist_median = grp["dist_ligne_km"].median()
        dist_p90    = grp["dist_ligne_km"].quantile(0.90)
        dist_p10    = grp["dist_ligne_km"].quantile(0.10)

        alt_sol_median = grp["alt_sol_m"].median()
        h_bs = MAT_HEIGHT_M.get(techno, 35.0)

        p_tx = P_TX_DBM.get(techno, 46.0)

        path_loss_med = select_path_loss_model(dist_median, freq_mhz, h_bs)
        snr_med = round(p_tx - path_loss_med - NOISE_FLOOR_DBM, 2)

        path_loss_p10 = select_path_loss_model(max(dist_p10, 0.1), freq_mhz, h_bs)
        snr_p10 = round(p_tx - path_loss_p10 - NOISE_FLOOR_DBM, 2)

        path_loss_p90 = select_path_loss_model(dist_p90, freq_mhz, h_bs)
        snr_p90 = round(p_tx - path_loss_p90 - NOISE_FLOOR_DBM, 2)

        bw = BW_MHZ_PAR_BANDE.get((techno, freq_mhz), BW_MHZ_DEFAULT.get(techno, 5.0))

        # [FIX-B2] Plafond réaliste par (techno, bande) au lieu de Shannon(30dB)
        debit_plafond = DEBIT_PLAFOND_REEL.get((techno, freq_mhz),
                        bw * math.log2(1 + 10**(20/10)))  # fallback Shannon(20dB)

        debit_max = shannon_mbps(snr_p10, bw, debit_plafond)
        debit_min = max(0.01, shannon_mbps(snr_p90, bw, debit_plafond))

        # [FIX-B1] RTT_RANGE corrigé pour 2G
        rtt_min, rtt_max = RTT_RANGE.get(techno, (50, 150))
        ber              = BER_BASE.get(techno, 1e-3)
        delta_snr        = DELTA_SNR.get(techno, 1.0)
        impact_vitesse   = IMPACT_VITESSE.get(techno, -0.10)
        portee           = PORTEE_KM.get((techno, freq_mhz), 5.0)

        key = f"{techno}_{freq_mhz}"
        profiles[key] = {
            "techno":            techno,
            "bande_mhz":         freq_mhz,
            "portee_km":         portee,
            "rtt_min_ms":        rtt_min,
            "rtt_max_ms":        rtt_max,
            "debit_max_mbps":    debit_max,
            "debit_min_mbps":    debit_min,
            "delta_snr_db":      delta_snr,
            "ber":               ber,
            "impact_vitesse":    impact_vitesse,
            "n_antennes_zone":   int(n_antennes),
            "dist_mediane_km":   round(dist_median, 2),
            "alt_sol_mediane_m": round(alt_sol_median, 1),
            "snr_median_db":     snr_med,
            "bw_mhz_reel":       bw,
            "p_tx_dbm":          p_tx,
            "debit_plafond_reel": debit_plafond,
            "modele_propagation": "COST231" if freq_mhz <= 2000 else "3GPP_TR38901_RMa",
            "source":            f"ANFR réel ({n_antennes} antennes zone Courpière-Ambert)",
        }

        stats.append({
            "techno": techno, "bande": freq_mhz, "n": n_antennes,
            "dist_med": round(dist_median, 1),
            "snr_med":  snr_med,
            "debit_max": debit_max,
            "debit_min": debit_min,
            "bw":        bw,
            "p_tx":      p_tx,
            "plafond":   debit_plafond,
            "modele":    "C231" if freq_mhz <= 2000 else "TR38901",
            "rtt":       f"{rtt_min}–{rtt_max}",
        })

    print(f"\n  {'Techno':<5} {'Bande':>6}  {'N':>3}  "
          f"{'Dist_med':>9}  {'SNR_med':>8}  "
          f"{'BW':>6}  {'Plafond':>8}  {'Débit_max':>10}  {'RTT':>12}  {'Modèle':>8}")
    print("  " + "-" * 105)
    for s in stats:
        print(f"  {s['techno']:<5} {s['bande']:>6}  {s['n']:>3}  "
              f"{s['dist_med']:>8.1f}km  "
              f"{s['snr_med']:>7.1f}dB  "
              f"{s['bw']:>5.0f}M  "
              f"{s['plafond']:>7.0f}M  "
              f"{s['debit_max']:>9.1f}M  "
              f"{s['rtt']:>12}  {s['modele']:>8}")

    return profiles


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("fetch_anfr_profiles.py — Profils depuis antennes ANFR réelles")
    print("  Zone : Courpière–Ambert, ligne 785000")
    print("  CORRECTIONS : A2 (modèle propagation), A3 (P_tx), A4 (BW réelle)")
    print("                B1 (RTT 2G), B2 (plafond débit 5G)")
    print("=" * 70)

    print("\n[1/4] Chargement données...")
    df_ligne = pd.read_csv(f"{INPUT_DIR}/ligne_gps.csv")

    dist_min_val = df_ligne["distance_km"].min()
    df_ligne_troncon = df_ligne[
        (df_ligne["distance_km"] >= dist_min_val) &
        (df_ligne["distance_km"] <= dist_min_val + 30.0)
    ].copy()

    import math as _math
    def hav(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = _math.radians(lat1), _math.radians(lat2)
        a = (_math.sin(_math.radians(lat2-lat1)/2)**2
             + _math.cos(phi1)*_math.cos(phi2)
             * _math.sin(_math.radians(lon2-lon1)/2)**2)
        return R * 2 * _math.asin(_math.sqrt(max(0, a)))

    troncon_pts = list(zip(
        df_ligne_troncon["lat"].values[::5],
        df_ligne_troncon["lon"].values[::5]
    ))

    df_ant_raw = pd.read_csv(f"{INPUT_DIR}/antennes_anfr.csv")
    if "coordonnees" in df_ant_raw.columns and "ant_lat" not in df_ant_raw.columns:
        def parse_coord(s):
            try:
                parts = str(s).replace(",", " ").split()
                return float(parts[0]), float(parts[1])
            except Exception:
                return None, None
        df_ant_raw[["ant_lat","ant_lon"]] = df_ant_raw["coordonnees"].apply(
            lambda s: pd.Series(parse_coord(s))
        )
        df_ant_raw = df_ant_raw.dropna(subset=["ant_lat","ant_lon"]).copy()

    print("  Calcul distances au tronçon Courpière-Ambert (0-30km)...")
    df_ant_raw["dist_troncon_km"] = df_ant_raw.apply(
        lambda r: min(hav(r["ant_lat"], r["ant_lon"], p[0], p[1])
                      for p in troncon_pts),
        axis=1
    )
    df_ant = df_ant_raw[df_ant_raw["dist_troncon_km"] <= 10.0].copy()

    LAT_MIN, LAT_MAX = 45.38, 46.10
    LON_MIN, LON_MAX = 3.30, 3.90
    mask_bbox = (
        (df_ant["ant_lat"] >= LAT_MIN) & (df_ant["ant_lat"] <= LAT_MAX) &
        (df_ant["ant_lon"] >= LON_MIN) & (df_ant["ant_lon"] <= LON_MAX)
    )
    df_ant = df_ant[mask_bbox].copy()
    df_ant["dist_ligne_km"] = df_ant["dist_troncon_km"]
    print(f"  Antennes ANFR zone Courpière-Ambert (≤10km + bbox) : {len(df_ant)}")
    print(f"  Tracé tronçon   : {len(df_ligne_troncon)} points")

    def norm_gen(g):
        g = str(g).strip().upper()
        if g in ("GSM","2G","EDGE"):        return "2G"
        if g in ("UMTS","3G","HSPA"):       return "3G"
        if g in ("LTE","4G","LTE-A"):       return "4G"
        if g in ("NR","5G","5GNR","NR5G"):  return "5G"
        return g
    df_ant["generation"] = df_ant["generation"].apply(norm_gen)
    df_ant = df_ant[df_ant["generation"].isin(["2G","3G","4G","5G"])].copy()

    print(f"  (techno, bande) : {df_ant.groupby(['generation','bande_mhz']).ngroups} combinaisons")
    print(f"\n  Par technologie :")
    for (gen, bande), grp in df_ant.groupby(["generation","bande_mhz"]):
        modele = "COST231" if int(bande) <= 2000 else "3GPP TR38901 RMa"
        print(f"    {gen}/{int(bande)} MHz : {len(grp)} antennes  "
              f"dist méd={grp['dist_troncon_km'].median():.1f} km  "
              f"modèle={modele}")

    print("\n[2/4] Altitudes EU-DEM des antennes...")
    df_ant["alt_sol_m"] = get_antenna_elevations(df_ant)
    print(f"  Altitudes antennes : "
          f"min={df_ant['alt_sol_m'].min():.0f}m  "
          f"médiane={df_ant['alt_sol_m'].median():.0f}m  "
          f"max={df_ant['alt_sol_m'].max():.0f}m")

    df_ant["dist_ligne_km"] = df_ant["dist_troncon_km"]
    profiles = compute_anfr_profiles(df_ant, df_ligne_troncon)

    print(f"\n[4/4] Sauvegarde → {OUTPUT_FILE}")
    output = {
        "meta": {
            "source":      "Antennes ANFR réelles zone Courpière-Ambert",
            "methode":     (
                "COST 231-Hata (f≤2000 MHz) + 3GPP TR 38.901 RMa (f>2000 MHz) "
                "avec P_tx réel par technologie et BW réelle ARCEP"
            ),
            "corrections": [
                "A2: modèle propagation", "A3: P_tx par techno", "A4: BW réelle",
                "B1: RTT 2G (300-700 ms)", "B2: plafond débit 5G réaliste",
            ],
            "generated_by":"fetch_anfr_profiles.py v3",
            "n_antennes":  int(len(df_ant)),
            "description": (
                "Profils technologiques calculés depuis les antennes ANFR "
                "réellement présentes dans la zone Courpière-Ambert (≤10km tronçon). "
                "Corrections v3 : RTT 2G plafonné (300-700 ms), débit 5G plafonné "
                "aux valeurs réalistes rurales (150/250/300 Mbps par bande)."
            )
        },
        "profiles": profiles,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✓ {OUTPUT_FILE} — {len(profiles)} profils sauvegardés")
    print("\n→ Prochaine étape : python prepare_data.py")
    print("  puis : python enrich_data.py --scenario ALL")


if __name__ == "__main__":
    main()