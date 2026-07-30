"""
fetch_pn_osm.py — Passages à niveau via OpenStreetMap
======================================================
Ce script récupère les passages à niveau sur la ligne
ferroviaire Courpière–Ambert depuis OpenStreetMap.

Pourquoi OpenStreetMap et pas SNCF ?
  L'API SNCF ne référence plus les passages à niveau des
  lignes fermées. OpenStreetMap, maintenu par la communauté,
  les conserve même sur les lignes hors service.

Source : Overpass API (https://overpass-api.de)
  Requête : noeuds tagués railway=level_crossing ou
  railway=crossing dans la bounding box de la ligne.

Sortie :
  data/raw/passages_niveau.csv
  Colonnes : osm_id, lat, lon, nom, type_pn

Usage :
  python fetch_pn_osm.py
"""

import os
import time

import pandas as pd
import requests

OUTPUT_FILE = "data/raw/passages_niveau.csv"
os.makedirs("data/raw", exist_ok=True)

# Bounding box du tronçon Courpière–Ambert
# Format Overpass : (lat_min, lon_min, lat_max, lon_max)
BBOX = "(45.53,3.53,45.76,3.76)"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_passages_niveau():
    """
    Interroge l'API Overpass pour récupérer tous les
    passages à niveau dans la bounding box de la ligne.

    Tags OSM utilisés :
      railway=level_crossing  — passage à niveau officiel
      railway=crossing        — croisement ferroviaire générique
    """

    # Requête Overpass QL
    # On cherche les deux types dans la bbox de la ligne
    query = f"""
[out:json][timeout:30];
(
  node["railway"="level_crossing"]{BBOX};
  node["railway"="crossing"]{BBOX};
);
out body;
"""

    print("  Appel Overpass API (OpenStreetMap)...")
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=40)
    r.raise_for_status()

    elements = r.json().get("elements", [])
    print(f"  {len(elements)} éléments OSM trouvés")

    if not elements:
        print("  ⚠ Aucun passage à niveau trouvé dans la bbox")
        return pd.DataFrame(columns=["osm_id", "lat", "lon", "nom", "type_pn"])

    rows = []
    for el in elements:
        tags = el.get("tags", {})
        rows.append({
            "osm_id":  el.get("id", ""),
            "lat":     el.get("lat"),
            "lon":     el.get("lon"),
            "nom":     tags.get("name", tags.get("ref", "")),
            "type_pn": tags.get("railway", ""),
        })

    df = pd.DataFrame(rows).dropna(subset=["lat", "lon"])
    return df


def main():
    print("=" * 60)
    print("fetch_pn_osm.py — Passages à niveau (OpenStreetMap)")
    print("=" * 60)

    df = fetch_passages_niveau()

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✓ {len(df)} passages à niveau sauvegardés")
    print(f"  → {OUTPUT_FILE}")
    if not df.empty:
        print(f"  Types : {df['type_pn'].value_counts().to_dict()}")
    print("\n→ Prochaine étape : python prepare_data.py")


if __name__ == "__main__":
    main()