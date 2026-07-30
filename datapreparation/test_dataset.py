import pandas as pd
import numpy as np

# Chargement du dataset
df = pd.read_csv('data/processed/dataset_enrichi.csv', low_memory=False)

# Vérification finale
print("=== VERIFICATION FINALE ===\n")

# Filtrage hors tunnel pour les générations réseau
cel = df[
    (df['generation'].isin(['2G', '3G', '4G', '5G'])) &
    (~df['in_tunnel'])
]

print("Packet loss (hors tunnel):")
print(
    cel.groupby('generation')['packet_loss_pct']
    .agg(['mean', 'median', 'max'])
    .round(4)
)

print("\nSat_conduite_distante par génération:")
print(
    df.groupby('generation')['sat_conduite_distante']
    .mean()
    .round(3)
)

print("\nFlag_zone_blanche:")
print(df['flag_zone_blanche'].value_counts().to_dict())
print("(flag_zone_blanche=0 est NORMAL: la 2G/3G/4G couvre toute la ligne hors tunnel)")

print("\nZone_blanche:")
print(df['zone_blanche'].value_counts().to_dict())

print("\nFlag_tunnel:")
print(df['flag_tunnel'].value_counts().to_dict())

print("\nScore QoS moyen:", round(df['score_qos'].mean(), 3))
print("Score QoS critique moyen:", round(df['score_qos_critique'].mean(), 3))
print("Score QoS standard moyen:", round(df['score_qos_standard'].mean(), 3))

print("\nDataset final:", df.shape)
print("Nombre de colonnes:", len(df.columns))