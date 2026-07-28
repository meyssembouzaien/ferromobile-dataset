# FerroMobile — AI-Driven Communication Intelligence for Autonomous Railway Vehicles

Research internship project at **UTBM / FEMTO-ST (DISC/OMNI)**, supervised by **Sassi Maaloul**.
M2 Thesis: *AI-Based Intelligent Connectivity for Autonomous Railway Vehicles*.

## Overview

FerroMobile designs an AI-driven decision system that keeps autonomous and
semi-autonomous railway vehicles reliably connected on secondary lines with
limited network coverage. The target corridor is an SNCF secondary line in
the Auvergne region, relying exclusively on public cellular networks
(2G/3G/4G/5G).

A moving train on this type of line is a demanding radio environment: hilly
and forested terrain causes frequent line-of-sight obstruction, tunnels
block external signal entirely, train speed degrades effective SNR, and
weather introduces additional variability. The system has to decide, in
real time, which available communication technology to use at each point
along the route, and where to place new antennas to close coverage gaps.

## Approach

- **Physics-first data pipeline**: builds a dataset pairing GPS points along
  the corridor with candidate antennas, using real terrain elevation, a
  standards-based propagation model (ITU-R P.1812) for path loss and
  line-of-sight testing, real historical weather data (ERA5 reanalysis)
  across four seasonal scenarios, and a standards-based rain attenuation
  model (ITU-R P.838-3).
- **QoS scoring**: throughput, latency, and bit-error-rate are each
  converted into a satisfaction score and combined into an overall
  quality-of-service label per (point, antenna) pair, used to rank
  candidate connectivity options. An earlier phase of the work explored
  conditioning antenna choice on per-application QoS profiles (e.g.
  video vs. telemetry vs. emergency braking); this was set aside in favor
  of a single generic QoS scoring approach that captures the
  throughput/latency/reliability trade-off relevant to critical rail
  operations, keeping the decision architecture simpler and directly
  actionable.
- **Machine learning**: a Learning-to-Rank model (XGBoost) handles
  real-time antenna/technology selection; a reinforcement learning agent
  is used for antenna placement optimization (where to add infrastructure
  to close coverage gaps).
- Every physical constant and acceptability threshold used in the pipeline
  is explicitly documented as either a standards-derived value or a
  transparent engineering choice, with its closest applicable reference —
  see `docs/FERROMOBILE_DATASET_DOCUMENTATION.md` for the full
  methodology.

## Pipeline

| Script | Purpose |
|---|---|
| `fetch_data.py` | Static infrastructure data: track geometry, tunnels, antennas, terrain |
| `fetch_temperature.py` | Historical weather data for the seasonal scenarios (ERA5 reanalysis) |
| `prepare_data.py` | Radio physics: path loss, line-of-sight, SNR, BER, throughput, RTT |
| `enrich_data.py` | Weather integration, QoS scoring, normalization, final dataset assembly |

Each step consumes the previous step's output; the full methodology,
formulas, and design rationale for every stage are documented in
`docs/FERROMOBILE_DATASET_DOCUMENTATION.md`.

## Repository structure

```
ferromobile-ai-connectivity/
├── data/                     # Raw and processed datasets (not versioned — see .gitignore)
├── src/
│   ├── fetch_data.py
│   ├── fetch_temperature.py
│   ├── prepare_data.py
│   └── enrich_data.py
├── notebooks/                 # EDA, classification, and model notebooks
├── docs/
│   └── FERROMOBILE_DATASET_DOCUMENTATION.md
├── .gitignore
└── README.md
```

## Status

Dataset pipeline finalized. Current focus: reviewing the state of the art
on antenna placement and selection approaches to position the
reinforcement-learning contribution of the thesis.

## Author

Meyssem Bouzaien — M2 IoT and Intelligent Systems, Faculty of Sciences of
Tunis. Research intern, UTBM / FEMTO-ST.
