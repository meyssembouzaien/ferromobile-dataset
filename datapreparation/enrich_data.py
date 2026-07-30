"""
enrich_data.py — Étape 3 : Enrichissement du dataset

Entrée  : data/processed/dataset_base.csv
          data/raw/temperature_scenarios.csv
Sortie  : data/processed/dataset_enrichi.csv

Étapes :
  1. Chargement + nettoyage
  2. Fusion météo par scénario
  3. Atténuation pluie (ITU-R P.838-3) sur débit et SNR
  4. Recalcul du BER après atténuation
  5. Scores QoS (Weber-Fechner + G.1030)
  6. Normalisations et flags
  7. Écriture

Lancer : python enrich_data.py --scenario ALL
"""

import argparse
import math
import os
import zlib

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════
# CHEMINS
# ═════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
INPUT_BASE  = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi.csv")
TEMPERATURE_CSV = os.path.join(BASE_DIR, "data", "raw", "temperature_scenarios.csv")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# ═════════════════════════════════════════════════════════════════
# CONSTANTES
# ═════════════════════════════════════════════════════════════════

# Heures représentatives par scénario (pic de pluie observé dans ERA5)
HEURE_PAR_SCENARIO = {"S1": 19, "S2": 19, "S3": 7, "S4": 5}

# ─── Radio ───
BW_SHANNON  = {"2G": 0.2, "3G": 5.0, "4G": 10.0, "5G": 100.0}   # MHz
SHANNON_EFF = 0.7
SNR_MAX_DB  = {"2G": 22.0, "3G": 28.0, "4G": 32.0, "5G": 35.0}
CODING_GAIN = {"2G": 2.0, "3G": 4.0, "4G": 6.0, "5G": 8.0}       # dB
BITS_PER_PKT = 400
PATH_FRAC   = 0.30   # fraction du lien sous la pluie

# ─── Coefficients ITU-R P.838-3 (polarisation horizontale) ───
P838_COEFFS = [
    (100, 0.0000387, 0.912), (400, 0.000154, 0.963),
    (700, 0.000352, 0.918), (900, 0.000650, 0.878),
    (1800, 0.00154, 0.841), (2100, 0.00188, 0.830),
    (2600, 0.00263, 0.811), (3500, 0.00489, 0.780),
    (5800, 0.0188, 0.689),  (12000, 0.0682, 0.629),
    (15000, 0.113, 0.599),
]

# ─── Seuils de couverture / débit par génération ───
DEBIT_VIABLE_MBPS = 1.0    # zone_blanche (couverture minimale)
DEBIT_MIN_PAR_GEN = {"2G": 0.05, "3G": 0.5, "4G": 5.0, "5G": 10.0}

# ─── QoS ───
# [1] Reichl et al., IEEE ICC 2010 — loi logarithmique (débit, BER)
# [2] ITU-T G.1030 (2014) — sigmoïde (latence)
# [3] ITU-T Y.1541 (2011) — seuils de latence
# [4] ARCEP, enquête qualité de service mobile — débit médian 5G France
DEBIT_SEUIL_MIN = 1.0     # Mbps
DEBIT_SEUIL_REF = 300.0   # Mbps — ARCEP médian 5G descendant [4]
RTT_SEUIL_MS    = 250.0
K_SIGMOID       = 0.02
BER_SEUIL_BON     = {"2G": 1e-2, "3G": 1e-3, "4G": 5e-4, "5G": 1e-4}
BER_SEUIL_MAUVAIS = {g: 10*v for g, v in BER_SEUIL_BON.items()}

# ─── Bruit stochastique pour réalisme ───
DEBIT_BRUIT_SIGMA = 0.15
DEBIT_BORNES = {"2G": (0.05, 0.24), "3G": (0.5, 7.2)}

# ─── Coûts d'infrastructure ───
INFRA_COSTS = {
    "2G":    (10_000,  100,  50),
    "3G":    (20_000,  200, 100),
    "4G":    (30_000,  300, 150),
    "4G_HB": (35_000,  350, 180),
    "5G":    (80_000,  800, 500),
}

# ─── Normalisation pour features ML ───
DEBIT_NORM_REF = 800.0     # Mbps — 3GPP TR 38.913 crête 5G NR


# ═════════════════════════════════════════════════════════════════
# FONCTIONS PHYSIQUES
# ═════════════════════════════════════════════════════════════════

def atten_pluie_p838(pluie_mm_h, freq_mhz):
    """Atténuation spécifique due à la pluie (dB/km). Réf : ITU-R P.838-3."""
    if pluie_mm_h <= 0:
        return 0.0
    freqs = [c[0] for c in P838_COEFFS]
    f = max(freqs[0], min(freqs[-1], freq_mhz))
    for i in range(len(freqs) - 1):
        f1, f2 = freqs[i], freqs[i+1]
        if f1 <= f <= f2:
            t = (math.log10(f) - math.log10(f1)) / (math.log10(f2) - math.log10(f1))
            k1, a1 = P838_COEFFS[i][1],   P838_COEFFS[i][2]
            k2, a2 = P838_COEFFS[i+1][1], P838_COEFFS[i+1][2]
            k = 10 ** (math.log10(k1) + t * (math.log10(k2) - math.log10(k1)))
            a = a1 + t * (a2 - a1)
            return round(k * (pluie_mm_h ** a), 5)
    return 0.0


def _erfc_approx(x):
    """Approximation numérique de erfc (Abramowitz & Stegun 7.1.26)."""
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    return (0.254829592*t - 0.284496736*t**2 + 1.421413741*t**3
            - 1.453152027*t**4 + 1.061405429*t**5) * math.exp(-x*x)


def ber_physique(snr_db, gen, rng):
    """BER analytique par modulation (Murota-Hirade 1981 pour GMSK)."""
    snr_lin = 10 ** (max(-10.0, min(50.0, snr_db)) / 10)
    if gen == "2G":
        ber = 0.5 * _erfc_approx(math.sqrt(max(0.0, 0.68 * snr_lin)))
    elif gen == "3G":
        ber = 0.5 * _erfc_approx(math.sqrt(max(0.0, snr_lin / 2)))
    elif gen == "4G":
        ber = (3/8) * _erfc_approx(math.sqrt(max(0.0, snr_lin / 10)))
    else:
        ber = (7/24) * _erfc_approx(math.sqrt(max(0.0, snr_lin / 42)))
    return round(max(1e-6, ber * (10 ** rng.normal(0.0, 0.3))), 12)


def get_infra_cost(gen, bande_mhz):
    """Retourne (cout_installation, cout_mensuel, conso_w)."""
    g = str(gen).upper()
    if "5G" in g:  return INFRA_COSTS["5G"]
    if "4G" in g or "LTE" in g:
        return INFRA_COSTS["4G_HB"] if float(bande_mhz) >= 1800 else INFRA_COSTS["4G"]
    if "3G" in g:  return INFRA_COSTS["3G"]
    return INFRA_COSTS["2G"]


# ═════════════════════════════════════════════════════════════════
# FONCTIONS DE SATISFACTION QoS
# ═════════════════════════════════════════════════════════════════

def qos_sat_debit(debit_mbps):
    """Satisfaction due au débit (Weber-Fechner, log)."""
    if debit_mbps is None or math.isnan(debit_mbps) or debit_mbps <= DEBIT_SEUIL_MIN:
        return 0.0
    if debit_mbps >= DEBIT_SEUIL_REF:
        return 1.0
    return round(
        (math.log(debit_mbps) - math.log(DEBIT_SEUIL_MIN)) /
        (math.log(DEBIT_SEUIL_REF) - math.log(DEBIT_SEUIL_MIN)), 4)


def qos_sat_latence(rtt_ms):
    """Satisfaction due au RTT (sigmoïde G.1030)."""
    if rtt_ms is None or math.isnan(rtt_ms):
        return 0.0
    return round(1.0 / (1.0 + math.exp(K_SIGMOID * (rtt_ms - RTT_SEUIL_MS))), 4)


def qos_sat_ber(ber, gen):
    """Satisfaction due au BER (log10, seuils par génération)."""
    if ber is None or math.isnan(ber) or ber <= 0:
        return 0.0
    g = str(gen).upper()
    bon     = BER_SEUIL_BON.get(g,     1e-3)
    mauvais = BER_SEUIL_MAUVAIS.get(g, 1e-2)
    if ber <= bon:     return 1.0
    if ber >= mauvais: return 0.0
    return round(1.0 - (math.log10(ber) - math.log10(bon)) /
                        (math.log10(mauvais) - math.log10(bon)), 4)


# ═════════════════════════════════════════════════════════════════
# MÉTÉO
# ═════════════════════════════════════════════════════════════════

def load_weather(df_pts, csv_path, scenario_id, heure):
    """Charge la météo ERA5 pour le scénario/heure demandé, fusionne sur les points."""
    df_sc = pd.read_csv(csv_path)
    df_hr = df_sc[(df_sc["scenario_id"] == scenario_id) & (df_sc["heure"] == heure)].copy()
    df_hr = df_hr.rename(columns={"temperature_c": "temp_c", "windspeed_kmh": "vent_kmh"})

    # Pluie horaire, avec fallback sur le max journalier si la valeur est 0
    df_day = df_sc[df_sc["scenario_id"] == scenario_id]
    df_max = (df_day.groupby("point_id")["precipitation_mm_h"].max()
                    .rename("pluie_max").reset_index())
    df_hr  = df_hr.rename(columns={"precipitation_mm_h": "pluie_mm_h"})
    df_hr  = df_hr.merge(df_max, on="point_id", how="left")
    df_hr["pluie_mm_h"] = df_hr["pluie_mm_h"].where(df_hr["pluie_mm_h"] > 0, df_hr["pluie_max"])

    # Pluie p95 = pic journalier
    df_p95 = (df_day.groupby("point_id")["precipitation_mm_h"].max()
                    .rename("pluie_p95_mm_h").reset_index())

    cols = ["point_id", "pluie_mm_h", "temp_c", "vent_kmh", "humidity_pct", "cloudcover_pct"]
    df = df_pts.merge(df_hr[cols], on="point_id", how="left")
    df = df.merge(df_p95, on="point_id", how="left")

    # Comble les trous par interpolation linéaire le long du tracé
    for col in ["pluie_mm_h", "pluie_p95_mm_h", "temp_c", "vent_kmh", "humidity_pct", "cloudcover_pct"]:
        df[col] = df[col].interpolate("linear").ffill().bfill()

    return df


# ═════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════

def main(scenario_id, heure):
    print("=" * 60)
    print(f"Enrichissement — scénario {scenario_id}, {heure}h")
    print("=" * 60)

    # ─── 1. Chargement ───
    df = pd.read_csv(INPUT_BASE, low_memory=False)
    df["ant_id"] = df["ant_id"].astype(str)
    print(f"[1] Dataset base : {len(df)} lignes")

    # Nettoyage sentinelles
    df.loc[df["rtt_ms"] == 9999, "rtt_ms"] = np.nan
    for c in ["snr_db", "snr_adjusted_db"]:
        df.loc[(df[c] == -99) & (~df["in_tunnel"]), c] = np.nan
    df["commune"] = df["commune"].fillna("INCONNU")

    # ─── 2. Météo ───
    df_pts = df[["point_id", "lat", "lon", "distance_km", "altitude_m",
                 "vegetation", "in_tunnel"]].drop_duplicates("point_id").reset_index(drop=True)
    df_meteo = load_weather(df_pts, TEMPERATURE_CSV, scenario_id, heure)
    print(f"[2] Météo : pluie moy = {df_meteo['pluie_mm_h'].mean():.3f} mm/h")

    # Perturbation spatiale (petit bruit reproductible)
    seed = zlib.crc32(scenario_id.encode()) % (2**16)
    rng  = np.random.default_rng(seed)
    df_meteo["pluie_mm_h"] = (df_meteo["pluie_mm_h"] * (10 ** rng.normal(0.0, 0.08, len(df_meteo))))
    df_meteo["pluie_mm_h"] = df_meteo["pluie_mm_h"].clip(lower=0).round(4)
    df_meteo["temp_c"]     = (df_meteo["temp_c"] + rng.normal(0.0, 0.4, len(df_meteo))).round(2)

    meteo_cols = ["point_id", "pluie_mm_h", "pluie_p95_mm_h", "temp_c",
                  "vent_kmh", "humidity_pct", "cloudcover_pct"]
    df = df.drop(columns=[c for c in meteo_cols[1:] if c in df.columns], errors="ignore")
    df = df.merge(df_meteo[meteo_cols], on="point_id", how="left")
    df["scenario_id"] = scenario_id

    # ─── 3. Atténuation pluie ───
    df["atten_pluie_db_km"] = df.apply(
        lambda r: atten_pluie_p838(r["pluie_mm_h"], float(r["bande_mhz"])), axis=1)

    def delta_snr_pluie(row):
        if row["in_tunnel"] or pd.isna(row["snr_adjusted_db"]):
            return 0.0
        return row["atten_pluie_db_km"] * row["dist_ant_km"] * PATH_FRAC

    df["_delta_snr"] = df.apply(delta_snr_pluie, axis=1)

    # Impact sur le débit via Shannon
    def new_debit(row):
        if row["in_tunnel"] or pd.isna(row["snr_adjusted_db"]):
            return row.get("debit_adjusted", 0.0)
        bw = BW_SHANNON.get(row["generation"], 10.0)
        snr_ref = 10 ** (max(-10.0, row["snr_adjusted_db"]) / 10)
        snr_eff = 10 ** (max(-10.0, row["snr_adjusted_db"] - row["_delta_snr"]) / 10)
        c_ref = bw * math.log2(1 + snr_ref) * SHANNON_EFF
        c_eff = bw * math.log2(1 + snr_eff) * SHANNON_EFF
        ratio = max(0.0, min(1.0, c_eff / c_ref)) if c_ref > 0 else 1.0
        return round(row["debit_adjusted"] * ratio, 4)

    df["debit_adj_mbps"] = df.apply(new_debit, axis=1).clip(lower=0)
    df["attenuation_pluie_pct"] = np.where(
        df["debit_adjusted"] > 0,
        (1 - df["debit_adj_mbps"] / df["debit_adjusted"]).clip(0, 0.8) * 100,
        0.0).round(4)

    # Bruit réaliste sur 2G/3G
    rng_bruit = np.random.default_rng(zlib.crc32((scenario_id + "_bruit").encode()))
    for gen, (dmin, dmax) in DEBIT_BORNES.items():
        mask = (df["generation"] == gen) & (~df["in_tunnel"]) & (df["debit_adj_mbps"] > 0)
        n = int(mask.sum())
        if n > 0:
            noise = rng_bruit.normal(0.0, DEBIT_BRUIT_SIGMA, n)
            df.loc[mask, "debit_adj_mbps"] = np.clip(
                df.loc[mask, "debit_adj_mbps"].values * (10 ** noise), dmin, dmax).round(4)

    # Débit sous pluie p95 (worst-case)
    def new_debit_p95(row):
        if row["in_tunnel"] or pd.isna(row["snr_adjusted_db"]):
            return row["debit_adj_mbps"]
        gamma95 = atten_pluie_p838(row["pluie_p95_mm_h"], float(row["bande_mhz"]))
        delta   = gamma95 * row["dist_ant_km"] * PATH_FRAC
        bw = BW_SHANNON.get(row["generation"], 10.0)
        snr_ref = 10 ** (max(-10.0, row["snr_adjusted_db"]) / 10)
        snr_eff = 10 ** (max(-10.0, row["snr_adjusted_db"] - delta) / 10)
        c_ref = bw * math.log2(1 + snr_ref) * SHANNON_EFF
        c_eff = bw * math.log2(1 + snr_eff) * SHANNON_EFF
        ratio = max(0.0, min(1.0, c_eff / c_ref)) if c_ref > 0 else 1.0
        return round(row["debit_adjusted"] * ratio, 4)

    df["debit_adj_p95_mbps"] = df.apply(new_debit_p95, axis=1).clip(lower=0)
    df["debit_adj_p95_mbps"] = df[["debit_adj_mbps", "debit_adj_p95_mbps"]].max(axis=1)

    # ─── 4. SNR + BER après pluie ───
    def new_snr(row):
        if row["in_tunnel"] or pd.isna(row["snr_adjusted_db"]):
            return row["snr_adjusted_db"]
        snr_new = row["snr_adjusted_db"] - row["_delta_snr"]
        cap = SNR_MAX_DB.get(row["generation"], 35.0)
        return round(min(max(snr_new, -10.0), cap), 3)

    df["snr_adjusted_db"] = df.apply(new_snr, axis=1)

    def new_ber(row):
        if row["in_tunnel"] or pd.isna(row["snr_adjusted_db"]):
            return row.get("ber")
        seed_row = zlib.crc32(f"{row['point_id']}_{row['ant_id']}_{scenario_id}".encode())
        rng_row  = np.random.default_rng(seed_row)
        return ber_physique(row["snr_adjusted_db"], row["generation"], rng_row)

    df["ber"] = df.apply(new_ber, axis=1)

    def new_pkt_loss(row):
        if row["in_tunnel"] or pd.isna(row["snr_adjusted_db"]):
            return row.get("packet_loss_pct")
        seed_row = zlib.crc32(f"{row['point_id']}_{row['ant_id']}_{scenario_id}_pkt".encode())
        rng_row  = np.random.default_rng(seed_row)
        snr_coded = row["snr_adjusted_db"] + CODING_GAIN.get(row["generation"], 0)
        ber_eff = ber_physique(snr_coded, row["generation"], rng_row)
        pc = math.exp(BITS_PER_PKT * math.log(max(1e-15, 1 - ber_eff)))
        return round(min(100.0, max(0.0, (1 - pc) * 100)), 6)

    df["packet_loss_pct"] = df.apply(new_pkt_loss, axis=1)
    df = df.drop(columns=["_delta_snr"])
    print(f"[3-4] Pluie appliquée, BER recalculé")

    # ─── 5. Coûts et QoS ───
    df[["cout_installation_eur", "cout_mensuel_eur", "conso_w"]] = df.apply(
        lambda r: pd.Series(get_infra_cost(r["generation"], r["bande_mhz"])), axis=1)

    df["qos_sat_debit"]   = df["debit_adj_mbps"].apply(qos_sat_debit)
    df["qos_sat_latence"] = df["rtt_ms"].apply(qos_sat_latence)
    df["qos_sat_ber"]     = df.apply(lambda r: qos_sat_ber(r["ber"], r["generation"]), axis=1)

    df["qos1"] = ((df["qos_sat_debit"] + df["qos_sat_latence"] + df["qos_sat_ber"]) / 3).round(4)
    df["qos2"] = (0.50 * df["qos_sat_debit"] +
                  0.30 * df["qos_sat_latence"] +
                  0.20 * df["qos_sat_ber"]).round(4)
    df["qos3"] = df["qos_sat_debit"].round(4)   # débit pur (focus critique)
    print(f"[5] QoS : qos1 moy = {df['qos1'].mean():.3f}, "
          f"qos2 moy = {df['qos2'].mean():.3f}, "
          f"qos3 moy = {df['qos3'].mean():.3f}")

    # ─── 6. Normalisations et flags ───
    df["debit_adj_norm"] = (df["debit_adj_mbps"] / DEBIT_NORM_REF).clip(0, 1).round(4)
    df["rtt_norm"]       = df.apply(
        lambda r: 1.0 if (pd.isna(r["rtt_ms"]) or r["in_tunnel"])
                  else round(min(1.0, r["rtt_ms"] / 500.0), 4), axis=1)
    df["cout_norm"]      = (df["cout_mensuel_eur"] / 800).clip(0, 1).round(4)
    df["dist_ant_norm"]  = (df["dist_ant_km"] / max(1.0, df["dist_ant_km"].max())).clip(0, 1).round(4)
    df["snr_norm"]       = df["snr_adjusted_db"].apply(
        lambda v: 0.0 if (v is None or (isinstance(v, float) and (math.isnan(v) or v <= -99.0)))
                  else round(max(0.0, min(1.0, v / 35.0)), 4))
    df["temp_norm"]      = ((df["temp_c"] + 10) / 50.0).clip(0, 1).round(4)
    df["pluie_norm"]     = (df["pluie_mm_h"] / 15.0).clip(0, 1).round(4)

    df["flag_no_signal"] = ((~df["in_tunnel"]) &
                            (df["snr_adjusted_db"].isna() | (df["snr_adjusted_db"] <= 10.0))
                           ).astype(int)

    # qualite_ok : portillon minimal par génération
    df["_seuil"] = df["generation"].map(DEBIT_MIN_PAR_GEN).fillna(3.0)
    df["qualite_ok"] = (
        (df["debit_adj_mbps"] >= df["_seuil"]) &
        (df["snr_adjusted_db"].notna()) & (df["snr_adjusted_db"] >= 5.0) &
        (df["rtt_ms"].notna()) & (df["rtt_ms"] <= 500.0) &
        (~df["in_tunnel"])
    )
    df = df.drop(columns=["_seuil"])

    # zone_blanche : aucun lien cellulaire hors tunnel n'atteint DEBIT_VIABLE_MBPS
    cell = df[df["generation"].isin(["2G","3G","4G","5G"]) & (~df["in_tunnel"])]
    best = cell.groupby("point_id")["debit_adj_mbps"].max()
    pts_hors_tun = df.loc[~df["in_tunnel"], "point_id"].unique()
    zb_map = {pid: int(best.get(pid, 0.0) < DEBIT_VIABLE_MBPS) for pid in pts_hors_tun}
    for pid in df.loc[df["in_tunnel"], "point_id"].unique():
        zb_map[pid] = 1
    df["zone_blanche"]      = df["point_id"].map(zb_map).fillna(1).astype(int)
    df["flag_zone_blanche"] = (df["zone_blanche"] == 1).astype(int)
    df["flag_debit_nul"]    = (df["debit_adj_mbps"] == 0).astype(int)

    df["tech_class"] = 1
    df["tech_label"] = "CELL"

    # Encodages numériques
    df["gen_num"] = df["generation"].map({"2G":1,"3G":2,"4G":3,"5G":4}).fillna(0).astype(int)
    df["veg_num"] = df["vegetation"].map({"plaine":0,"urbain":1,"foret_legere":2,"foret_dense":3}).fillna(0).astype(int)

    # Nettoyage colonnes redondantes
    df = df.drop(columns=[c for c in ["debit_adjusted", "delta_ber", "pkt_loss_norm"]
                          if c in df.columns])

    # ─── 7. Écriture ───
    df = df.sort_values(["point_id", "generation", "bande_mhz"]).reset_index(drop=True)
    df["row_id"] = range(len(df))
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[7] Écrit : {OUTPUT_FILE}")
    print(f"    {len(df)} lignes × {len(df.columns)} colonnes\n")


# ═════════════════════════════════════════════════════════════════
# ENTRÉE
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="S3",
                        choices=["S1", "S2", "S3", "S4", "ALL"])
    parser.add_argument("--heure", type=int, default=None)
    args = parser.parse_args()

    if args.scenario == "ALL":
        all_dfs = []
        for sc in ["S1", "S2", "S3", "S4"]:
            main(sc, args.heure or HEURE_PAR_SCENARIO[sc])
            all_dfs.append(pd.read_csv(OUTPUT_FILE, low_memory=False))
        combined = pd.concat(all_dfs, ignore_index=True)
        combined["row_id"] = range(len(combined))
        combined.to_csv(OUTPUT_FILE, index=False)
        print(f"Multi-scénarios : {len(combined)} lignes")
        print(combined["scenario_id"].value_counts().to_dict())
    else:
        heure = args.heure or HEURE_PAR_SCENARIO[args.scenario]
        main(args.scenario, heure)