"""
clean_cache.py — Nettoyage du cache météo et fichiers incomplets
================================================================
À lancer AVANT fetch_data.py quand tu as eu des interruptions (Ctrl+C)
ou des erreurs 429 consécutives.

Usage :
  python3 clean_cache.py
"""

import os
import json

DATA_RAW = "data/raw"

# Fichiers à supprimer pour repartir proprement
FILES_TO_CLEAN = [
    f"{DATA_RAW}/meteo_clusters_cache.json",   # cache météo partiel
    f"{DATA_RAW}/meteo_par_point.csv",          # météo interpolée incomplète
    f"{DATA_RAW}/meteo_clusters.csv",           # clusters bruts incomplets
    f"{DATA_RAW}/meteo_historique.csv",         # centroïde incomplet
]

print("=" * 50)
print("clean_cache.py — Nettoyage cache météo")
print("=" * 50)

for path in FILES_TO_CLEAN:
    if os.path.exists(path):
        os.remove(path)
        print(f"  ✓ Supprimé : {path}")
    else:
        print(f"  — Absent   : {path}")

print()
print("✓ Nettoyage terminé.")
print("→ Tu peux maintenant relancer : python3 fetch_data.py")
print("  (Attends 2-3h si Open-Meteo était bloqué)")
