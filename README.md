# FerroMobile Dataset — Data Pipeline for Autonomous Railway Communication Intelligence

Research internship project at **UTBM / FEMTO-ST (DISC/OMNI)**, supervised by **Sassi Maaloul**.
M2 Thesis: *AI-Based Intelligent Connectivity for Autonomous Railway Vehicles*.

This repository contains the complete data pipeline that builds the FerroMobile
dataset: GPS geometry, terrain, weather, and cellular network data for an SNCF
secondary line, enriched with radio-physics simulation and QoS scoring.

Model training and validation live in a separate repository:
[`ferromobile-model-training`](https://github.com/meyssembouzaien/ferromobile-model-training),
to keep this repo focused on data and that one lightweight.

## Overview

FerroMobile designs an AI-driven decision system that keeps autonomous and
semi-autonomous railway vehicles reliably connected on secondary lines with
limited network coverage, relying exclusively on public cellular networks
(2G/3G/4G/5G). This repository covers the first half of the project: turning
raw geographic, weather, and network-registry data into a physics-grounded,
QoS-scored dataset ready for machine learning.

## Repository structure

```
ferromobile-dataset/
├── data/                # Raw source data and final processed datasets
│   ├── raw/              # GPS trace, tunnels, ANFR antennas, weather, terrain
│   └── processed/         # dataset_base.csv, dataset_enrichi.csv — the deliverables
├── dataprepataion/       # Pipeline scripts, tests, and dataset-construction notebooks
│   ├── fetch_data.py
│   ├── fetch_temperature.py
│   ├── fetch_anfr_profiles.py
│   ├── fetch_pn_osm.py
│   ├── prepare_data.py
│   ├── enrich_data.py
│   ├── clean_cache.py
│   ├── test_dataset.py / test_dates_s3.py / test.py
│   └── *.ipynb            # LOS, SNR, threshold-tuning, zone-blanche notebooks
├── EDA/                  # Exploratory data analysis on the final dataset
│   └── eda_dataset_enrichi_final.ipynb
├── docs/
│   └── FERROMOBILE_DATASET_DOCUMENTATION.md   # Full methodology reference
├── requirements.txt
└── .gitignore
```

Note: `data/cache/` and cache JSON files are intentionally excluded from
version control (see `.gitignore`) — they hold no information beyond what the
fetch scripts regenerate automatically. `crc_covlib`/`crc-covlib` (the
third-party radio propagation library) and `venv/` are also excluded and
should be reinstalled locally.

## Pipeline

| Script | Purpose |
|---|---|
| `dataprepataion/fetch_data.py` | Static infrastructure data: track geometry, tunnels, antennas, terrain |
| `dataprepataion/fetch_temperature.py` | Historical weather data for the seasonal scenarios (ERA5 reanalysis) |
| `dataprepataion/fetch_anfr_profiles.py` | ANFR antenna technology profile enrichment |
| `dataprepataion/fetch_pn_osm.py` | Level-crossing data (OpenStreetMap fallback source) |
| `dataprepataion/prepare_data.py` | Radio physics: path loss, line-of-sight, SNR, BER, throughput, RTT |
| `dataprepataion/enrich_data.py` | Weather integration, QoS scoring, normalization, final assembly |
| `dataprepataion/clean_cache.py` | Utility to clear stale fetch caches |

Full methodology, every formula, threshold, and design rationale is documented
in `docs/FERROMOBILE_DATASET_DOCUMENTATION.md`.

## Setup

```bash
pip install -r requirements.txt
```

`crc-covlib` (compiled radio propagation library) is a hard dependency of
`prepare_data.py` and is not versioned in this repo — install it separately
per its own instructions.

## Reproduction

See `docs/FERROMOBILE_DATASET_DOCUMENTATION.md`, Section 10, for the full
step-by-step reproduction guide and prerequisites.

## Status

Dataset pipeline finalized. Current focus (in the companion training repo):
reviewing the state of the art on antenna placement and selection approaches
to position the reinforcement-learning contribution of the thesis.

## Author

Meyssem Bouzaien — M2 IoT and Intelligent Systems, Faculty of Sciences of
Tunis. Research intern, UTBM / FEMTO-ST.
