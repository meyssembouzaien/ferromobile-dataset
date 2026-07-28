# FerroMobile Dataset — Complete Technical Documentation

**Project:** AI-driven intelligent communication system for autonomous/semi-autonomous railway vehicles
**Line:** SNCF secondary line 785000, Courpière–Ambert (~38.6 km, Auvergne, France)
**Institution:** FEMTO-ST (UTBM, DISC/OMNI department)
**Scope of this document:** the complete offline data pipeline that produces `dataset_enrichi.csv`, the dataset used for antenna-selection (Learning-to-Rank) and, downstream, antenna-placement (Reinforcement Learning) work.

## How to use this document

This document is written so that a reader with a background in telecommunications, signal processing, and data engineering can reproduce the dataset from scratch without reading the source code. Every formula, every threshold, every design decision, and every fallback path is documented. Where a value is an engineering choice rather than a value dictated by a standard, this is stated explicitly — the goal is that nothing is silently assumed to be "the truth" when it is actually a modeling decision.

The pipeline has four sequential scripts, run in this order:

1. `fetch_data.py` → static infrastructure data (track, tunnels, antennas, terrain)
2. `fetch_temperature.py` → weather scenario data (ERA5 reanalysis)
3. `prepare_data.py` → radio physics: path loss, LOS, SNR, BER, throughput, RTT
4. `enrich_data.py` → weather integration, QoS scoring, normalization, final assembly

Each script reads from `data/raw/` or `data/processed/` and writes back to those same directories. The final deliverable is `data/processed/dataset_enrichi.csv`.

## Table of Contents

1. Project Objective and Design Philosophy
2. Pipeline Architecture — High-Level Flow
3. Step 1 — Static Data Collection (fetch_data.py)
4. Step 1b — Weather Data Collection (fetch_temperature.py)
5. Step 2 — Radio Physics Computation (prepare_data.py)
6. Step 3 — Weather Integration and QoS Scoring (enrich_data.py)
7. Reproducibility and Determinism Strategy
8. Complete Column Dictionary — Final Dataset
9. Known Limitations, Assumptions, and Engineering Choices
10. Step-by-Step Reproduction Guide
11. References

---

## 1. Project Objective and Design Philosophy

### 1.1 What this dataset is for

The dataset models, for every 20-meter point along the Courpière–Ambert railway corridor, the radio-link quality that every nearby cellular antenna (2G/3G/4G/5G) would offer to a train travelling at a reference speed, under four distinct seasonal weather conditions. It exists to support two downstream machine-learning tasks:

- **Antenna selection** (static, per point): given several candidate antennas at a given location, which one should the train's communication system connect to? This is framed as a Learning-to-Rank (LTR) problem — the dataset provides, for each (point, antenna) pair, a set of physical features and a scalar quality label (`qos1` / `qos2`) that acts as the ranking target.
- **Antenna placement** (future work, Reinforcement Learning): given the coverage gaps and QoS levels observed in this dataset, where should new antennas be installed to minimize CAPEX while satisfying the QoS requirements of critical train applications?

### 1.2 Why this is hard, physically

A moving train is one of the most demanding radio environments for cellular connectivity:

- **Terrain**: the Livradois-Forez region is hilly and forested, causing frequent Line-of-Sight (LOS) obstruction.
- **Tunnels**: 12 tunnels totaling several kilometers, where no external signal reaches the train at all.
- **Speed**: at typical regional-rail speeds, Doppler-related and channel-estimation effects degrade the effective SNR compared to a stationary receiver.
- **Vegetation**: dense forest attenuates signal differently than open plains or urban clutter.
- **Weather**: rain theoretically attenuates radio signals (this dataset explicitly models and then quantifies this effect — see Section 6.4 for why the measured effect turns out to be physically small at the frequencies used here).

### 1.3 Design philosophy: physics first, engineering choices declared

Two categories of numbers appear throughout this pipeline, and the distinction between them matters for anyone trying to defend or reproduce this work:

- **Physically-sourced constants**: numbers taken directly from an ITU-R Recommendation, a 3GPP Technical Report, a peer-reviewed paper, or a national regulator's published methodology. These are cited by name and document number everywhere they appear.
- **Engineering choices**: numbers that had to be picked because no single standard prescribes them for this exact use case (e.g., "what throughput counts as a minimally viable service for 2G on this specific corridor?"). These are documented as engineering choices, with the closest applicable reference used as an anchor — never presented as if they were normative values.

This distinction is maintained consistently through the rest of this document.

---

## 2. Pipeline Architecture — High-Level Flow

```
                     ┌─────────────────────────┐
                      │      fetch_data.py       │
                      │  (Step 1 — static data)  │
                      └────────────┬─────────────┘
                                   │
    ┌──────────────────┬──────────┼──────────────┬─────────────────────┐
    ▼                  ▼          ▼               ▼                     ▼
ligne_gps.csv   passages_niveau  tunnels.csv  antennes_anfr.csv   terrain/*.tif
(GPS trace,       .csv (info      (12 tunnels,   (2G/3G/4G/5G       (local DEM
 resampled,        only, not      embedded       antennas within    raster for
 with altitude)    used in        list)          10 km corridor)    LOS testing)
                   radio calc)

                      ┌─────────────────────────┐
                      │  fetch_temperature.py    │
                      │ (Step 1b — weather data)  │
                      └────────────┬─────────────┘
                                   │
                                   ▼
                    temperature_scenarios.csv
                 (4 seasonal scenarios × 24 hours
                  × spatial clusters, ERA5 reanalysis)

 ligne_gps.csv, tunnels.csv,          ┌─────────────────────────┐
 antennes_anfr.csv, terrain/*.tif ───▶│      prepare_data.py     │
                                      │ (Step 2 — radio physics) │
                                      └────────────┬─────────────┘
                                                   │
                                                   ▼
                                     data/processed/dataset_base.csv
                                (point × antenna pairs, LOS-filtered,
                                 path loss, SNR, BER, throughput, RTT
                                 — computed for ONE reference weather-
                                 free, rain-free condition, at 80 km/h)

 dataset_base.csv,                    ┌─────────────────────────┐
 temperature_scenarios.csv ──────────▶│      enrich_data.py      │
                                      │ (Step 3 — weather +      │
                                      │  QoS scoring + assembly) │
                                      └────────────┬─────────────┘
                                                   │
                                                   ▼
                                data/processed/dataset_enrichi.csv
                              (FINAL DATASET — 4 weather scenarios ×
                               all point-antenna pairs, QoS-scored,
                               normalized, ready for LTR / RL)
```

### 2.1 Why the pipeline is split into two physics stages

`prepare_data.py` computes the "dry, reference-weather" radio physics once (path loss, LOS, base SNR, base throughput). `enrich_data.py` then applies the weather scenario on top of that base computation, four times (once per season), and derives the QoS scores. This split exists for a practical reason: path loss computation via the ITU-R P.1812 propagation model (see Section 5.6) is by far the most computationally expensive step in the whole pipeline (it involves querying a compiled C++ library, `crc-covlib`, per point-antenna pair, with terrain profile sampling). Recomputing it four times — once per weather scenario — would be wasteful, since weather does not change the path loss geometry, only the received signal's degradation on top of that fixed geometry. So the expensive step runs once, and the cheap weather-dependent adjustments run four times.

---

## 3. Step 1 — Static Data Collection (fetch_data.py)

This script produces every static (non-weather) input the rest of the pipeline needs. It has six sub-steps.

### 3.1 Track trace (GPS points)

**Source**: a local CSV file (`data/raw/courpiere_ambert_resampled.csv`) containing the raw GPS trace of the line, expected to have `lat`/`lon` columns (also accepts `latitude`/`longitude` variants).

**Processing**:

- The trace is validated (must have ≥ 2 valid GPS points, columns must exist after rename normalization).
- Orientation check: the script computes the great-circle (haversine) distance from the trace's first and last point to the fixed coordinate of Courpière station (45.7651° N, 3.5402° E). If the last point is closer to Courpière than the first point, the whole trace is reversed, guaranteeing a consistent Courpière → Ambert direction for the `distance_km` column downstream.
- Sanity check: the raw trace length (sum of haversine distances between consecutive points) is expected to fall between 28 and 50 km; a warning (not a hard failure) is printed otherwise, since the real line length is ~38.6 km.
- Resampling: the trace is resampled to a constant spacing of `SPACING_M = 100` meters using linear interpolation along the polyline (see Section 3.1.1 for the algorithm).

**Output**: `ligne_gps.csv` with columns `point_id`, `lat`, `lon`, `distance_km` (altitude is added later, in sub-step 3.5).

#### 3.1.1 Linear resampling algorithm (interpolate_line)

Given a sequence of (lat, lon) waypoints, the function walks along each segment between consecutive raw waypoints and drops a new point every `spacing_m` meters of cumulative distance, using linear interpolation in (lat, lon) space (not true great-circle interpolation — acceptable given the very short segment lengths involved, on the order of 100 m).

For a segment from (lat0, lon0) to (lat1, lon1) of length `seg_m` meters:

```
f = pos / seg_m                    # fraction along the segment, pos = distance already walked into this segment
lat_p = lat0 + f * (lat1 - lat0)
lon_p = lon0 + f * (lon1 - lon0)
```

A `carry_m` variable tracks leftover distance from one segment to the next so that resampled points remain evenly spaced across segment boundaries (no gap or overlap at waypoint junctions). Each new point's `distance_km` is computed as the previous point's `distance_km` plus the haversine distance to the new point (not the naive linear-space distance), so that the final `distance_km` column reflects true great-circle cumulative distance.

Note on resampling granularity: this initial resampling in `fetch_data.py` uses 100 m spacing. `prepare_data.py` (Section 5) later re-resamples again to a finer 20 m spacing — the 100 m trace here is only an intermediate product, not the final spatial resolution of the dataset.

### 3.2 Level crossings (passages_niveau.csv)

**Source, primary**: SNCF Open Data API (ressources.data.sncf.com), dataset `liste-des-passages-a-niveau`, filtered by `code_ligne='785000'`. Pagination is handled automatically (100 records per page).

**Source, fallback**: if the SNCF API is unreachable, an Overpass (OpenStreetMap) query is issued instead, against a hardcoded bounding box covering the Courpière–Ambert area (45.53,3.53,45.76,3.76), searching for `railway=level_crossing` and `railway=crossing` tags. Four public Overpass mirrors are tried in sequence in case of failure.

**Post-processing**: every retrieved crossing is filtered to keep only those within `RAYON_KM = 10.0` km of the resampled trace (distance computed via a KD-tree built on the trace's (lat, lon) coordinates — see Section 3.6.1 for how the KD-tree distance function works).

Note: this file is collected for completeness / potential future use but is not consumed by `prepare_data.py` or `enrich_data.py` in the current pipeline — it plays no role in the radio physics or QoS computation documented in this file.

### 3.3 Tunnels (tunnels.csv)

**Source**: a hardcoded, manually verified list of 12 tunnels on this stretch of line, embedded directly in the script (not fetched from any API). Each entry specifies the tunnel's name, length in meters, and the GPS coordinates of both its entrance and exit portals.

| Tunnel | Length (m) |
|---|---|
| Sauviat | 302 |
| Archimbaud | 142 |
| Cublas | 193 |
| Graves | 319 |
| Saint Gervais | 124 |
| Constancis | 46 |
| Olliergues | 117 |
| Chalard | 241 |
| Got | 61 |
| Flouvat | 168 |
| Châtelet | 586 |
| Perrier | 238 |

Why embedded and not fetched: no public API reliably provides tunnel geometry for this specific secondary line at the precision needed (portal-to-portal coordinates). These were manually verified once and hardcoded for reproducibility — every re-run of the pipeline uses exactly the same 12 tunnels, with no dependency on an external service that could change or disappear.

### 3.4 Mobile network antennas (ANFR)

**Source**: the French national frequency agency (ANFR — Agence Nationale des Fréquences) open dataset `observatoire_2g_3g_4g`, queried via data.anfr.fr, for the two administrative departments covering the corridor (dept 063 = Puy-de-Dôme, dept 043 = Haute-Loire).

**Processing**:

- All records for both departments are fetched (paginated, 100 records per request).
- Deduplication by `recordid`.
- Status filter: only antennas with `statut` in {"En service", "Techniquement opérationnel", "Approuvé"} are kept — antennas that are planned but not yet built, decommissioned, or in an unclear administrative state are excluded.
- Coordinate parsing: the ANFR `coordonnees` field (a single string like "lat, lon") is split and parsed; records that fail to parse are dropped.
- Frequency band extraction: the antenna's operating band in MHz is extracted from the free-text `emr_lb_systeme` field by scanning its whitespace-separated tokens for an integer in the plausible cellular range [100, 40000] MHz.
- Corridor filter: only antennas within `RAYON_KM = 10.0` km of the resampled trace are kept (same KD-tree distance method as level crossings).

**Output columns**: `ant_id`, `support_id`, `operateur`, `generation`, `bande_mhz`, `systeme`, `statut`, `ant_lat`, `ant_lon`, `commune`, `code_insee`, `dist_ligne_km`.

Important caveat for reproducibility: the ANFR database is a live, regularly-updated public registry. Re-running this fetch on a different date will not necessarily produce the same set of antennas — operators add, decommission, or reclassify sites over time. If exact antenna-set reproducibility is required, the CSV output of a specific run must be archived and reused directly, rather than re-fetched.

### 3.5 Point-wise elevation (EU-DEM 25 m, via OpenTopoData)

**Source**: the OpenTopoData public API (api.opentopodata.org/v1/eudem25m), serving the Copernicus EU-DEM 25-meter digital elevation model.

**Processing**: for every resampled trace point, the elevation is queried in batches of `TOPO_BATCH = 100` points per request, with bilinear interpolation requested from the API. If a batch fails (network error, unexpected API status, or a response with a different point count than requested), every point in that batch is set to `None` and a warning is printed — there is no retry at this specific step (contrast this with the local DEM raster construction below, which does retry — see 3.6.2).

Post-processing (in `main()`): any `None` elevation is filled via linear interpolation along the `point_id` sequence (`pandas.interpolate("linear")`), then rounded to 1 decimal. This assumes elevation varies smoothly enough between consecutive 100 m-spaced points that linear interpolation is an acceptable approximation for isolated API failures — it is not a substitute for systematic elevation data.

**Output**: the `altitude_m` column added to `ligne_gps.csv`.

### 3.6 Local DEM raster (for Line-of-Sight testing in prepare_data.py)

This is the most involved sub-step of `fetch_data.py`, and it exists to solve a specific performance problem.

#### 3.6.1 Why a local raster instead of per-point API calls

`prepare_data.py`'s Line-of-Sight (LOS) test (Section 5.5) needs the terrain elevation profile between every candidate antenna and every trace point — not just elevation at the trace points themselves. Sampled at up to 12 points per km along each sight line (capped at 400 samples per link), across tens of thousands of point-antenna candidate pairs, this would require querying the OpenTopoData API hundreds of thousands of times — far beyond the API's rate limit (1 request/second) and completely impractical in terms of runtime.

The solution: build one local elevation raster (a GeoTIFF grid) that covers the entire study corridor once, at the start of the pipeline, and have `prepare_data.py` read elevation values directly from this local file (near-instant) instead of hitting the API for every LOS sample.

Utility functions supporting this:

```
haversine_km(lat1, lon1, lat2, lon2)
# Great-circle distance in km between two GPS points:
#   a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
#   distance = 2·R·arcsin(√a),   R = 6371.0 km (mean Earth radius)

build_kdtree(pts)
# Builds a scipy cKDTree on the (lat, lon) coordinates of the resampled trace,
# for fast nearest-neighbor lookups (used to filter level crossings and antennas
# to those within RAYON_KM of the line — the actual haversine distance is
# recomputed exactly for the nearest neighbor found by the tree, since the
# tree itself operates on unprojected lat/lon and its raw distances are not
# metric-accurate).
```

#### 3.6.2 Raster construction algorithm

- Bounding box: compute the bounding box of all trace points, then expand it by `RAYON_KM + DEM_BUFFER_MARGIN_KM = 10.0 + 2.0 = 12.0` km in every direction (converted from km to degrees using the standard 111 km/degree-of-latitude approximation, with a longitude correction by `cos(mean_latitude)` to account for meridian convergence).
- Grid resolution: `DEM_GRID_SPACING_M = 100` meters — chosen to be consistent with the original 100 m trace resampling and more than sufficient for a LOS test that samples the terrain profile at ~83 m intervals (12 samples/km).
- Query: the grid (n_rows × n_cols points, north-to-south, west-to-east ordering) is queried against OpenTopoData in batches of 100 points, with up to `DEM_BATCH_MAX_RETRIES = 3` retries per batch (5-second delay between retries) — unlike the point-wise elevation fetch in 3.5, this step retries aggressively because a hole in the raster would silently corrupt every LOS test that happens to sample that location later.
- Failure threshold: if, after all retries, more than `DEM_MAX_NAN_FRACTION = 15%` of the grid remains unfilled, the entire raster is rejected (the function returns `None`, and the raster is not written). This is a deliberate design choice: writing a raster that is mostly nearest-neighbor-filled would silently reproduce the exact "elevation cache masks the real bug" failure mode that this local raster was built to prevent in the first place (see Section 9 for the historical context of that earlier bug).
- Gap filling: if the NaN fraction is below the 15% threshold, remaining gaps are filled using `scipy.interpolate.griddata` with `method="nearest"` (nearest-neighbor fill on the grid indices, not a smooth interpolation) — acceptable for a small fraction of scattered failures, not acceptable as the primary filling strategy.
- Write and verify: the raster is written as a GeoTIFF (EPSG:4326, single float32 band, `nodata=NaN`) to `data/raw/terrain/eu_dem_courpiere_ambert.tif`. After writing, the file is immediately reopened and a 1×1 pixel window is read back to confirm the file is not empty and not corrupted — a fail-fast check against silent write failures (disk full, permission issues, etc.).

---

## 4. Step 1b — Weather Data Collection (fetch_temperature.py)

### 4.1 Rationale for using real, hourly weather data

An annual or monthly climatological average would smooth away exactly the kind of variability (day/night cycles, discrete rain events) that matters for evaluating a system meant to be robust to varied conditions. Instead, this script fetches real, complete 24-hour weather histories for specific calendar days, from the ERA5 reanalysis dataset — a physically consistent, no-synthetic-model source.

### 4.2 Seasonal scenario selection

Four representative days from real historical data are used, one per season. The dates were specifically selected for containing meaningful precipitation, since the whole point of having weather scenarios is to have contrast between them:

| Scenario | Season | Date | Selection rationale |
|---|---|---|---|
| S1 | Winter | 2023-12-10 | Persistent winter rain |
| S2 | Spring | 2023-04-19 | Moderate spring rain |
| S3 | Summer | 2023-06-30 | Intense convective thunderstorm — the most extreme rain scenario in the set |
| S4 | Autumn | 2023-10-20 | Autumn rain episode with strong wind |

Selection methodology: representative days per season, selected among ERA5 archive days meeting a minimum precipitation threshold, chosen to maximize weather contrast rather than to represent a "statistically typical" day. A companion script, `test_dates_s3.py`, was used during development to scan candidate dates and report their max/mean hourly precipitation before the final four dates were locked in — this script is a data-exploration utility, not part of the production pipeline.

Reference hour per scenario: for each scenario, `enrich_data.py` (Section 6) extracts a single representative hour of the day — chosen as the hour of peak precipitation observed in that scenario's 24-hour ERA5 record.

### 4.3 Spatial clustering and API querying

Querying the ERA5 archive at every single one of the trace points, for four dates, would be both unnecessary (ERA5's native spatial resolution is ~31 km, far coarser than the corridor's point spacing) and wasteful of API quota. Instead:

- The corridor is divided into spatial clusters using `build_clusters()`: walking along the trace, a new cluster centroid is created every time the cumulative distance since the last cluster reaches `CLUSTER_KM = 10.0` km — a resolution deliberately chosen to be consistent with ERA5's native ~31 km grid.
- For each (cluster, scenario_date) pair, the Open-Meteo Archive API (archive-api.open-meteo.com/v1/archive, ERA5 reanalysis model) is queried for the full 24-hour record of: `temperature_2m`, `precipitation`, `windspeed_10m`, `relativehumidity_2m`, `cloudcover`.
- Caching: every successful query is cached to `data/raw/temperature_cache.json`, keyed by (lat, lon, date) rounded to 4 decimals — so an interrupted run can resume without re-querying already-fetched data.
- Rate limiting: a fixed `API_DELAY_S = 2.5` second delay between requests, plus exponential backoff (30s → 60s → 120s) specifically on HTTP 429 (rate limit) responses.
- Missing values within a fetched day: any NaN in the 24-hour record for a given variable is filled by linear interpolation along the hour axis, then forward/backward-filled at the edges.

### 4.4 Spatial interpolation from clusters to trace points

For each scenario and each hour of the day (0–23), the cluster-level values of each weather variable are interpolated to every trace point using `numpy.interp` — simple 1-D linear interpolation along cumulative `distance_km`, not a 2-D spatial interpolation. This is a deliberate simplification: since the trace is a 1-D corridor and clusters are ordered along it, treating the interpolation as 1-D (distance-along-track vs. weather value) is both simpler and arguably more physically meaningful than a 2-D geographic interpolation, since it inherently follows the corridor's actual path through varying terrain rather than interpolating "as the crow flies."

### 4.5 Altitude correction for temperature

ERA5 temperature values are reported at the elevation of the queried cluster centroid, which will generally differ from the elevation of the specific trace point being assigned that value (since points within a cluster can span a real elevation range in hilly terrain). A standard atmospheric lapse-rate correction is applied:

```
T_corrected = T_cluster + (−0.0065 °C/m) × (altitude_point − altitude_cluster)
```

Reference: −6.5 °C per 1000 m is the ICAO/WMO International Standard Atmosphere adiabatic lapse rate (1975).

### 4.6 Output

`data/raw/temperature_scenarios.csv`, with columns: `point_id`, `lat`, `lon`, `distance_km`, `altitude_m`, `scenario_id`, `saison`, `date`, `heure`, `temperature_c`, `precipitation_mm_h`, `windspeed_kmh`, `humidity_pct`, `cloudcover_pct`.

---

## 5. Step 2 — Radio Physics Computation (prepare_data.py)

This is the core physics engine of the pipeline. It runs once, at a fixed reference speed and with no weather effects applied — weather is layered on top of this output later, in `enrich_data.py`. Output: `data/processed/dataset_base.csv`.

### 5.1 Mandatory dependency: crc-covlib and ITU-R P.1812

The script hard-depends on `crc-covlib`, a compiled C++ radio propagation library (Communications Research Centre Canada), used exclusively for its implementation of the ITU-R P.1812 point-to-area propagation model for terrestrial services in the 30 MHz – 6 GHz range.

This is a deliberate, non-negotiable design choice: the script performs a self-test at import time (computing path loss for a known transmitter/receiver pair and checking the result falls in a plausible range) and exits immediately if `crc-covlib` is unavailable or the self-test fails. There is no analytical fallback model (no Hata, no COST-231, no free-space approximation). The rationale: a silently-degraded path loss model would produce a dataset whose quality varies depending on which machine or environment it was generated on, without any indication in the data itself that this happened — an unacceptable reproducibility risk for a thesis-grade dataset.

### 5.2 Corridor filtering and re-resampling

- Corridor cutoff: trace points beyond `TRONCON_KM_MAX = 40.0` km (from the start of `ligne_gps.csv`) are dropped, and `distance_km` is re-zeroed to the corridor's actual start.
- Re-resampling: the (already 100 m-resampled) trace from `fetch_data.py` is resampled again, this time to a finer default spacing — using the exact same linear-interpolation-along-polyline logic described in Section 3.1.1. This is the spacing that defines the dataset's final spatial granularity.

### 5.3 Tunnel marking

Using the 12 embedded tunnel entrance/exit coordinates (Section 3.3), every trace point is flagged `in_tunnel = True` if it falls within the entrance–exit span plus a `TUNNEL_BUFFER_M = 50` meter buffer on each side (via nearest-point lookup on a KD-tree built on the resampled trace, converting the tunnel portal coordinates to a position along the cumulative-distance axis). The buffer accounts for the fact that signal degradation at a tunnel mouth begins slightly before the physical portal, not abruptly at it.

### 5.4 Vegetation / land cover classification

**Primary source**: ESA WorldCover, a global 10 m-resolution land cover raster, if a local raster file is present in `data/raw/landcover/`. Each trace point's (lat, lon) is used to sample the raster value at that pixel, which is then mapped from the ESA WorldCover class codes to one of four simplified categories used throughout the rest of the pipeline: `foret_dense`, `foret_legere`, `plaine`, `urbain`.

**Fallback** (if no local raster available): a coarse heuristic based purely on `distance_km` along the corridor, calibrated by manual inspection of this specific line's known character. This fallback is explicitly a corridor-specific heuristic, not a general-purpose model — it should never be reused for a different line without re-calibration.

### 5.5 Antenna preprocessing

- Generation normalization: raw ANFR generation labels are mapped to a canonical set — GSM/2G/EDGE → "2G", UMTS/3G/HSPA → "3G", LTE/4G/LTE-A → "4G", NR/5G/5GNR/NR5G → "5G". Antennas outside {2G,3G,4G,5G} are dropped.
- Coordinate parsing: the ANFR `coordonnees` field is re-parsed (comma-or-space separated) into `ant_lat`/`ant_lon`.
- Range assignment (`portee_km`): each antenna is assigned a maximum theoretical range, looked up from the `TECH_PROFILES` table (keyed by generation and band), used later purely to build the initial candidate-antenna search radius — not as a hard propagation limit.
- Antenna mast height: each antenna's elevation is first read from the local DEM raster (or the OpenTopoData fallback) at the antenna's exact coordinates, then a technology-specific mast height is added on top (mast heights decrease from 2G through 5G, reflecting that newer deployments on this corridor tend to be smaller-cell, lower-mounted sites relative to legacy macro towers — an engineering assumption reflecting typical French rural deployment patterns, not a value taken from a specific standard).

### 5.6 Point–antenna association (candidate generation)

Testing every point against every antenna would be wasteful, since most pairs are far beyond any plausible range. The association is built in two filtering stages:

- **Corridor buffer (coarse filter)**: a `CORRIDOR_RAYON_KM = 10.0` km buffer polygon is built around the full trace (computed in the metric EPSG:2154 projection — Lambert-93, the standard French metric CRS — to get an accurate buffer in meters, then reprojected back to WGS84). Only antennas whose location intersects this buffer are considered candidates at all.
- **Per-antenna range + Top-K cap (fine filter)**: for each surviving antenna, all trace points within its range are found via a KD-tree query on the trace, and true haversine distances are computed. Then, for each (point, generation) combination, only the top-K closest antennas of that generation are kept, with the cap increasing from 2G through 5G. This cap exists to keep the dataset size tractable while still preserving enough antenna diversity per point for a meaningful ranking problem.

### 5.7 Technology profile table (TECH_PROFILES)

This is the single most important lookup table in the pipeline — it supplies, per (generation, band), the parameters that drive throughput and RTT calculation (Section 5.9) and, indirectly, the BER threshold reused later in `enrich_data.py`'s QoS scoring.

Columns in the table include: range, RTT min–max, peak/minimum throughput, speed-impact in dB, reference BER, a stochastic Δ-SNR noise term, and the throughput interpolation profile shape (linear for most technologies, sigmoid-shaped for the higher-band 5G profiles, reflecting a more realistic "cliff-edge" throughput drop-off characteristic of high-frequency, narrow-beam 5G NR deployments compared to the gentler linear degradation of lower-frequency, wider-coverage legacy technologies).

Provenance: these values were originally set from published bibliographic figures for each technology/band combination (3GPP specifications, ARCEP measurement campaigns, vendor datasheets for typical macro-cell deployments). They represent a synthesis rather than a single citable source per cell of the table — this should be stated plainly in any defense of this work, rather than over-claiming a single authoritative source for the whole table.

### 5.8 Line-of-Sight (LOS) testing

Every candidate (point, antenna) pair that survives the association step (Section 5.6) is subjected to a binary LOS test before any radio physics is computed. Pairs that fail this test are eliminated entirely — there is no NLOS (non-line-of-sight) propagation model in this pipeline; a blocked path simply removes that candidate from the dataset for that point.

#### 5.8.1 Algorithm

- If the antenna is within 50 m of the point, LOS is trivially assumed True (avoids degenerate sampling at near-zero distance).
- Otherwise, the straight line between antenna and point is sampled at roughly 12 samples per km of link distance, with a floor for very short links and a cap for very long ones.
- Terrain elevation is queried at every sample point (local DEM raster first, falling back to the OpenTopoData API, then to nearest-trace-point elevation as a last resort — see Section 5.8.3).
- Effective Earth curvature correction: because a real radio path is not a straight line relative to the curved Earth's surface, an effective-Earth-radius correction is added to the terrain elevation profile, using the standard "4/3 Earth radius" model used throughout radio engineering to approximate the bending of radio waves due to atmospheric refraction.
- Clearance check: a straight sight-line is drawn from (antenna elevation + mast height) to (point elevation), and LOS is declared True only if every intermediate terrain sample (curvature-corrected) stays at or below this line — a strict geometric clearance test, with no additional Fresnel-zone clearance margin applied.

#### 5.8.2 Result: high candidate elimination

On the actual corridor run, a majority of candidate (point, antenna) pairs generated by the association step were eliminated as NLOS, reflecting the genuinely hilly, forested character of the Dore valley corridor.

#### 5.8.3 Elevation lookup priority

Terrain elevation, used both for LOS sampling and for antenna mast-base elevation, is resolved through a three-level priority: local DEM raster (fast, primary source), OpenTopoData API (used only for points not covered by the local raster), and nearest-trace-point elevation (an absolute last-resort fallback if both fail). An in-memory + on-disk JSON cache avoids re-querying the same rounded coordinate twice across the whole run.

#### 5.8.4 Tunnels bypass LOS testing entirely

For any point flagged `in_tunnel = True` (Section 5.3), the LOS test is skipped altogether: `los_category` is forced to 0 (the sentinel value meaning "no external signal," as opposed to 1 for a valid LOS link), and radio metrics are set to sentinel values directly (Section 5.9) — a tunnel physically blocks all external cellular signal by construction, so there is no propagation physics to compute.

### 5.9 Path loss (ITU-R P.1812) and SNR

For every LOS-validated (point, antenna) link:

#### 5.9.1 Path loss

The `crc-covlib` library is called with the antenna's location and mast height as transmitter, the antenna's band (MHz) as frequency, a receiver height representative of an antenna mounted on a train carriage roof, the ITU_R_P_1812 propagation model at median/typical-case propagation statistics, terrain elevation from the local DEM raster, and a land-cover clutter category mapped from the point's vegetation classification (Section 5.4).

Sanity bounds: any resulting path loss outside a plausible dB range, or a NaN/Inf result, is treated as an invalid link and the pair is dropped from the dataset — same treatment as an eliminated NLOS link, with no analytical fallback, consistent with the "physics or nothing" philosophy stated in Section 5.1.

#### 5.9.2 SNR from path loss

SNR is derived from transmit power, path loss, and a standard thermal noise floor (a physical constant, not a tunable parameter), using a fixed receiver noise figure assumption applied uniformly across all technologies, and per-generation noise bandwidth and transmit power values. A per-technology SNR ceiling is then applied, representing the practical maximum SNR a real receiver of that generation can usefully exploit, beyond which additional signal strength does not translate into additional link quality because the receiver's own hardware becomes the limiting factor, not the channel.

#### 5.9.3 Speed effect on SNR — Doppler / channel-estimation degradation

At train speeds above a threshold, a speed-dependent SNR degradation is applied on top of the geometry-based SNR, using a per-(generation, band) impact factor from the `TECH_PROFILES` table — larger for narrowband legacy technologies. This term is a documented engineering model of Doppler shift and channel-estimation error at speed, not a citation-backed formula from a specific standard.

Sign convention (important, and previously a source of a real bug in this codebase): speed must subtract from SNR, never add — a faster train cannot physically improve link quality. An earlier version of this script had the sign reversed, which caused SNR values to exceed each technology's own physical ceiling. The corrected formula both subtracts and re-applies the ceiling after the subtraction, guaranteeing the result never exceeds the technology's SNR ceiling regardless of the speed term.

### 5.10 Throughput and RTT

Two alternative throughput/RTT models exist in the code, selected via a command-line argument. The default, and the one used to produce the documented dataset, is the empirical model.

#### 5.10.1 Empirical model (default)

Throughput is interpolated between the technology's close-range peak and far-range minimum (from `TECH_PROFILES`), as a function of link distance and range, using either a linear profile (used by 2G, 3G, and lower-band 4G/5G) or a sigmoid profile (used by the two higher-frequency 5G bands, reflecting a sharper "cliff-edge" throughput drop typical of higher-frequency 5G NR coverage). RTT is linearly interpolated between the technology's min and max RTT as a function of the distance ratio to maximum range.

#### 5.10.2 Alternative Shannon-capacity model (not used for the documented dataset)

For completeness, an alternative computes throughput from Shannon capacity with a standard engineering derating factor representing the gap between theoretical capacity and real-world achievable throughput. This model is documented for completeness but was not the one used to generate the reference dataset.

### 5.11 Tunnel sentinel values

For any point flagged `in_tunnel = True`, no radio physics is computed at all — sentinel values are assigned directly (no path loss, worst-case BER, zero throughput, a sentinel SNR value meaning "no usable signal," and `path_loss_model = "TUNNEL"`). The SNR sentinel is later converted to a proper NaN in post-processing, so that downstream numeric aggregations do not silently treat it as a real signal measurement.

### 5.12 Bit Error Rate (BER)

BER is computed analytically from the actual per-link adjusted SNR (not looked up from the `TECH_PROFILES` reference table, which instead only anchors the QoS acceptability threshold downstream — Section 6.6), using modulation-specific coherent-detection error-probability approximations, referenced to Murota & Hirade (1981) for 2G GMSK and standard coherent-detection approximations for 3G/4G/5G.

A stochastic realism noise term perturbs the deterministic formula (seeded deterministically, to avoid a dataset where BER is a perfectly deterministic function of SNR alone — real-world BER measurements always show scatter around the theoretical curve due to unmodeled effects), with a numerical floor applied below which differences in BER are not considered meaningful for this application.

Packet loss is then derived from a coded BER (applying a per-generation forward-error-correction coding gain before recomputing it) and an assumed packet size — an explicitly documented engineering assumption representing a typical small telemetry/control message size, not a cited standard packet size. The coding-gain values are order-of-magnitude figures consistent with typical coding gains reported in the digital communications literature, but are not tied to a specific per-generation standard either.

### 5.13 Stochastic augmentation columns — computed but not applied downstream

Two additional columns are computed for every link, explicitly documented in the source code as generic stochastic noise terms for data augmentation — intended to add realistic-scale variability to training data, not a physical model of any real-world effect, and never fed back into any other calculation in either `prepare_data.py` or `enrich_data.py`. These are explicitly not a model of rain/weather attenuation, and not a re-application of the speed-based SNR degradation. Anyone using this dataset should either put them to active use or explicitly document them as unused, rather than assume they already encode some physical effect they do not encode.

### 5.14 Output assembly and qualite_ok / zone_blanche

For every valid (point, antenna) link, one row is written to `dataset_base.csv`. Points with no valid link at all get an explicit placeholder row, so that every original trace point is represented in the dataset even when zero antennas cover it — essential for correctly computing zone-blanche (coverage-gap) statistics later, rather than silently dropping uncovered points.

`qualite_ok` (a binary per-link gate, computed here using a single generic threshold across all generations — this is later replaced by a per-generation threshold version computed in `enrich_data.py`, Section 6.9): combines a minimum throughput, minimum SNR, maximum RTT, and non-tunnel condition. `zone_blanche` (per point): a point is a "white zone" (coverage gap) if no link at that point satisfies `qualite_ok`. Tunnel points are always flagged `zone_blanche = 1` by construction.

### 5.15 Categorical numeric encodings

Two convenience integer-encoded columns are added for ML consumption, in addition to the string categorical columns: `gen_num` (generation encoding) and `veg_num` (vegetation category encoding).

---

## 6. Step 3 — Weather Integration and QoS Scoring (enrich_data.py)

This script takes `dataset_base.csv` (the weather-free, single-speed radio physics output of Section 5) and produces the final `dataset_enrichi.csv`, run once per weather scenario, then concatenated.

### 6.1 Initial cleaning

Before anything else, defensive cleaning passes run on the loaded `dataset_base.csv`: legacy sentinel RTT values are replaced with NaN; SNR sentinel values outside tunnels are replaced with NaN (inside tunnels, the sentinel is correct and expected); packet-loss sentinel values inside tunnels are replaced with NaN (a tunnel's packet loss is meaningless/undefined). Missing commune values (a genuine gap in the ANFR source data, not a pipeline bug) are filled with an explicit placeholder string rather than left as NaN, specifically because an empty string written to CSV silently becomes NaN again on re-read, which would hide the gap instead of documenting it.

### 6.2 Weather scenario merge

For the requested scenario, the corresponding hour's weather record (Section 4.2's reference-hour table) is read from `temperature_scenarios.csv` and merged onto every trace point.

A fallback stochastic weather generator exists as a development/testing safety net if the weather file is missing, using a fixed monthly climatology table with Gaussian-kernel spatial smoothing and a log-normal rain-occurrence model — this fallback was not the path used to produce the documented dataset (which used real ERA5 data) and should not be relied upon for any dataset intended for scientific reporting.

### 6.3 Reproducible spatial perturbation noise

A small amount of spatially-uncorrelated noise is added on top of the interpolated weather values, to avoid every point sharing an artificially identical value within a cluster.

**Determinism**: the random generator for this step is seeded via `zlib.crc32(scenario_id.encode())` — not Python's built-in `hash()` function. This distinction matters: Python's native `hash()` for strings is randomized on every process start (a security feature, `PYTHONHASHSEED`), so a seed derived from `hash(scenario_id)` would silently differ between two separate runs of the script, producing a different (but internally consistent) dataset each time — defeating reproducibility without any visible error. `zlib.crc32` is a pure, unsalted hash function, guaranteeing the exact same seed, and therefore the exact same noise values, on every run, on any machine.

### 6.4 Rain attenuation model — ITU-R P.838-3

This is the physical core of the weather effect. It is applied consistently across three quantities — throughput, SNR, and (indirectly, via SNR) BER and packet loss — a deliberate correction from an earlier pipeline version that applied rain attenuation to throughput only, which was physically inconsistent (two linked quantities, SNR and throughput, reacting differently to the same rain).

#### 6.4.1 Specific attenuation coefficient

ITU-R P.838-3 models rain-induced specific attenuation (dB per km of path length under rain) as a function of rain rate and frequency-dependent coefficients, log-log interpolated from the standard's reference table (horizontal polarization).

#### 6.4.2 Applying attenuation to throughput

Only the fraction of the point-to-antenna path assumed to actually be under active rainfall is attenuated — an explicit engineering assumption (not every kilometer between antenna and receiver is necessarily under the same rain cell), documented as such. Throughput loss is derived via a Shannon-ratio method comparing capacity with and without the rain-induced SNR loss. A parallel calculation using each scenario-day's peak (worst-hour) precipitation produces a companion "what if peak rain intensity applied continuously" column.

#### 6.4.3 Applying attenuation to SNR (and, through it, BER)

The rain-induced SNR loss is subtracted directly from the adjusted SNR, then re-capped at that generation's SNR ceiling. BER and packet loss are then recomputed from scratch from this rain-adjusted SNR, using the exact same modulation-specific formulas as in `prepare_data.py` (Section 5.12) — duplicated in this file rather than imported, to avoid a circular import dependency between the two scripts when either is run standalone.

**Determinism of the BER re-roll**: each row's BER re-computation uses its own random generator, seeded from a deterministic hash of (point_id, ant_id, scenario_id) via `zlib.crc32` — guaranteeing that the same link, in the same weather scenario, always gets the same stochastically-perturbed BER value across repeated runs of the pipeline, while still allowing genuine row-to-row variability.

#### 6.4.4 The physical result: rain attenuation at these frequencies is small

Applying this exact chain even to the most extreme rain scenario in the set yields a rain-induced SNR loss that is negligible compared to a typical link's SNR margin. This is not a modeling flaw — it is the correct, expected physical behavior of rain attenuation at sub-6 GHz cellular frequencies. ITU-R P.838-3 itself shows this: its attenuation coefficient grows by roughly three orders of magnitude between the lower and upper end of the cellular frequency range. Rain fade only becomes operationally significant well above the frequencies used on this corridor (relevant to satellite links and mmWave 5G bands, not the sub-4 GHz macro-cellular bands used here). This should be stated explicitly as a finding, not hidden as an inconvenient result: the dataset correctly shows that weather has a measurable-but-small effect on throughput and essentially no effect on the binary link-quality gate (`qualite_ok`, Section 6.9) at these frequencies.

### 6.5 Throughput noise for legacy technologies (2G/3G)

A log-normal multiplicative noise, applied only to non-tunnel 2G/3G links with non-zero rain-adjusted throughput, then clipped to technology-specific bounds. Rationale: 2G/3G, being narrowband, exhibit throughput that is essentially flat with distance in this pipeline's empirical model, which would otherwise produce an unrealistically deterministic-looking throughput column for these two generations specifically. This noise term is only applied to 2G/3G — 4G/5G throughput already has enough natural variability from the distance-based interpolation and rain attenuation. Determinism: seeded via a scenario-specific hash.

### 6.6 Infrastructure cost lookup

Used for potential CAPEX-aware reward shaping in downstream RL work (not used in the QoS scores themselves): install cost, monthly cost, and power consumption per generation/band. These are engineering placeholder figures representative of typical French rural macro-cell deployment costs, not sourced from a specific published cost study — documented here as such for full transparency.

### 6.7 QoS satisfaction functions

Three piecewise-linear (or log-linear, for BER) "satisfaction" functions map a raw physical quantity to a [0, 1] score, each independently, before being combined into the final `qos1`/`qos2` labels.

#### 6.7.1 Throughput satisfaction

A piecewise function scoring throughput against a threshold anchored between the ARCEP-published quality-of-service survey thresholds for "basic web browsing" and "streaming" service categories, chosen as the closest ARCEP-anchored value to a generic "minimally usable" mobile data service. This is not itself an ARCEP-mandated threshold — it is an engineering choice anchored to the closest applicable ARCEP reference point.

#### 6.7.2 Latency satisfaction

RTT is scaled per generation (legacy technologies inherently tolerate/require a different latency budget than modern ones), referenced to ERA/ERTMS/033281 and 3GPP TR 38.913 — these documents establish that the acceptability threshold should be scaled per generation rather than applying one fixed millisecond threshold uniformly. A rain-inflation factor is applied to the effective RTT used in this satisfaction score, anchored empirically by field measurements reporting a moderate RTT increase under intense LTE precipitation conditions, with the chosen factor set as a conservative engineering choice above this empirical range rather than a direct field-measured value for this specific corridor.

Important implementation detail: this is the only place in the entire pipeline where rain affects RTT — and it does so only inside this satisfaction-function calculation. The stored `rtt_ms` column itself is never modified by weather — it remains exactly the same value across all four weather scenarios for a given (point, antenna) pair (this is a factual, verifiable property of the dataset: raw `rtt_ms` is weather-invariant by construction, while `qos_sat_latence` is not).

#### 6.7.3 BER satisfaction

A log-linear interpolation between a "good" and "bad" BER threshold per generation (the "bad" threshold set at ten times the "good" one). Reference: the "good" threshold per generation is the same reference BER value from the `TECH_PROFILES` table (Section 5.7), itself anchored to Murota & Hirade (1981) for 2G and standard modulation-specific approximations for 3G/4G/5G.

### 6.8 Final QoS scores — qos1 and qos2

Two scoring variants are computed: an equal-weighted mean of the three satisfaction dimensions (`qos1`), and a throughput-weighted combination giving more weight to throughput than to latency or BER (`qos2`) — reflecting a generic-usage assumption that raw data capacity matters somewhat more than the other two dimensions for a broad, non-application-specific service quality metric. Neither is "more correct" than the other — they represent two different, explicitly-stated hypotheses about what matters for a generic (non-application-profile-conditioned) quality score. An empirical comparison on this dataset shows the two agree closely on the best-ranked antenna at a given point, and always agree at the generation level (2G vs 3G vs 4G vs 5G) — the practical difference between using one or the other as an LTR training label is very small on this corridor.

**Documented limitation — circularity**: both `qos1` and `qos2` are linear (or near-linear) combinations of feature columns (`debit_adj_mbps`, `rtt_ms`, `ber`) that are also present in the dataset as independent features. If a downstream model is trained using both the raw physical features and `qos1`/`qos2` as inputs to predict something derived from `qos1`/`qos2` again, this constitutes label leakage. This should be stated transparently in any report using this dataset, and the QoS-score columns should be excluded from any feature set whose target is itself QoS-derived.

### 6.9 qualite_ok and zone_blanche, recomputed

Important distinction from Section 5.14: the `qualite_ok` computed in `prepare_data.py` used a single generic throughput threshold across all generations. The version computed here, in `enrich_data.py`, and which is the one present in the final `dataset_enrichi.csv`, uses a per-generation throughput threshold instead, each anchored to the closest applicable reference (ETSI TS 145.001 for 2G, 3GPP TS 25.101 for 3G, the ARCEP survey thresholds for 4G, and 3GPP TR 38.913's eMBB target for 5G, scaled down to a realistic rural floor). The SNR and RTT gates use a generic digital-demodulation viability threshold (consistent with standard treatments in Proakis, *Digital Communications*) and the most permissive ITU-T Y.1541 latency class respectively, chosen because this gate is meant to represent "does any usable service exist at all," not "is this link good enough for a time-critical application."

All of the above thresholds are explicitly engineering choices, anchored to the closest applicable reference, not values mandated by any single standard for this exact use case. This should be presented in any defense of this work as: "thresholds set in consistency with the referenced literature, as a minimal-service floor per generation," not as normative values.

**What `qualite_ok` is for, and what it is not**: this is a coarse, binary coverage gate — used to compute `zone_blanche` and as an internal consistency check during development. It is not a QoS score, and is not intended to be used as a training feature or reward signal — that role belongs to `qos1`/`qos2` (Section 6.8), which are continuous and carry far more information. A documented, verified property of the current dataset: because `snr_adjusted_db` and raw `rtt_ms` are essentially weather-invariant, and the rain-driven change in throughput is small (Section 6.4.4), `qualite_ok` never flips value for the same (point, antenna) pair across the four weather scenarios in the documented run — this is a real, verified, and expected property, not a bug.

`zone_blanche`, per point, uses a separate, simpler threshold on throughput alone (not the full `qualite_ok` gate) — a point is a white zone if the best available cellular link there (of any generation) fails to reach even this minimal viable throughput, set explicitly below the lowest ARCEP-referenced threshold, on the reasoning that below this floor there is no usable data service at all, regardless of latency or error rate. Because this uses a different threshold from `qualite_ok`, the `zone_blanche` percentage computed here can differ slightly from a coverage statistic computed directly off `prepare_data.py`'s own `qualite_ok`-based `zone_blanche` (Section 5.14) — both are legitimate but answer subtly different questions ("is there any signal at all" vs. "is there signal that clears the full per-generation quality bar"), and should not be conflated when reporting coverage statistics.

### 6.10 Normalization features (for ML consumption)

A set of [0, 1]-scaled companion columns is added, for direct use as neural-network / RL state-space inputs without further scaling work downstream: normalized throughput (anchored to the 3GPP TR 38.913 5G NR peak reference), RTT, monthly cost (anchored to the most expensive tier), distance to antenna (dataset-relative), SNR (anchored to the highest technology SNR ceiling), and temperature.

### 6.11 Flags

Convenience binary flags are added: `flag_no_signal` (practically no usable signal outside a tunnel), `flag_zone_blanche` (a redundant mirror of `zone_blanche`), and `flag_debit_nul` (zero throughput).

### 6.12 Final assembly and column cleanup

Redundant columns superseded by others retained in the final dataset are dropped before final write. Rows are sorted and a fresh sequential `row_id` is assigned.

An internal consistency check is performed automatically at the end of each scenario run, verifying that no row is flagged `qualite_ok = True` while falling below its own generation-specific throughput threshold — a sanity assertion on the per-generation threshold logic, printed but not hard-enforced (a violation would print a non-zero count rather than raise an exception).

Multi-scenario concatenation: the script is run once per scenario (S1, S2, S3, S4), each producing a full `dataset_enrichi.csv`, and the four outputs are concatenated (`row_id` reassigned sequentially over the combined table) into the final multi-scenario dataset.

---

## 7. Reproducibility and Determinism Strategy

Every stochastic element in the pipeline is seeded deterministically:

| Stochastic element | Location | Seed derivation | Granularity |
|---|---|---|---|
| BER realism noise (base, weather-free) | prepare_data.py | fixed literal seed | one shared stream, whole run |
| Δ-SNR / Δ-speed augmentation terms | prepare_data.py | fixed literal seeds | one shared stream, whole run |
| Spatial weather perturbation noise | enrich_data.py | `zlib.crc32(scenario_id)` | one stream per scenario |
| 2G/3G throughput noise | enrich_data.py | `zlib.crc32(scenario_id + suffix)` | one stream per scenario |
| Rain-adjusted BER re-roll | enrich_data.py | `zlib.crc32(point_id, ant_id, scenario_id)` | one stream per row |
| Rain-adjusted packet-loss re-roll | enrich_data.py | same, with a suffix | one stream per row |

**Why `zlib.crc32` and not Python's built-in `hash()`**: Python randomizes the hash of `str` objects on every process start by default (`PYTHONHASHSEED`), as a security mitigation against hash-flooding denial-of-service attacks on dictionaries. This means `hash("S1")` produces a different integer every time the script is launched, even though the string is identical — so any seed derived from `hash(...)` would silently differ between two runs of the pipeline, producing a different but internally self-consistent dataset each time, with no visible error or warning. `zlib.crc32` is a pure checksum function with no such randomization; the same input string always produces the same integer, on any machine, in any process. Using `hash()` for seeding anywhere in a reproducibility-critical pipeline is a bug, not a style choice — this was identified and corrected during this project's development.

**Verifying reproducibility in practice**: running the full pipeline twice in succession and comparing a checksum of the output file confirms determinism:

```bash
python enrich_data.py --scenario ALL
md5sum data/processed/dataset_enrichi.csv > run1.md5
python enrich_data.py --scenario ALL
md5sum data/processed/dataset_enrichi.csv > run2.md5
diff run1.md5 run2.md5     # should report no difference
```

Note that this check assumes the inputs (`dataset_base.csv`, `temperature_scenarios.csv`) are themselves unchanged between the two runs — `enrich_data.py`'s determinism does not extend to `prepare_data.py`'s inputs (the ANFR antenna fetch, in particular, is not guaranteed reproducible across time — see Section 3.4's caveat) or to external API data sources in general.

---

## 8. Complete Column Dictionary — Final Dataset

`dataset_enrichi.csv` — one row per (point, antenna, weather scenario) link.

### 8.1 Identifiers and geometry

| Column | Type | Description |
|---|---|---|
| `row_id` | int | Sequential unique row identifier, reassigned after final multi-scenario concatenation |
| `point_id` | int | Identifier of the trace point |
| `lat`, `lon` | float | GPS coordinates of the trace point |
| `distance_km` | float | Cumulative distance from the Courpière end of the corridor |
| `altitude_m` | float | Trace point elevation (m), from EU-DEM |
| `vegetation` | str | One of `plaine`, `urbain`, `foret_legere`, `foret_dense` (Section 5.4) |
| `veg_num` | int | Integer encoding of vegetation (Section 5.15) |
| `in_tunnel` | bool | Whether this point falls within a tunnel + buffer (Section 5.3) |

### 8.2 Antenna identity and link geometry

| Column | Type | Description |
|---|---|---|
| `ant_id` | str | ANFR antenna record identifier (empty/NaN for orphan points with no candidate — Section 5.14) |
| `operateur` | str | Mobile network operator |
| `generation` | str | 2G, 3G, 4G, or 5G |
| `gen_num` | int | Integer encoding of generation (Section 5.15) |
| `bande_mhz` | int | Antenna operating frequency band |
| `commune` | str | Municipality of the antenna site (placeholder string if missing from ANFR source — Section 6.1) |
| `ant_lat`, `ant_lon` | float | Antenna GPS coordinates |
| `ant_alt_abs_m` | float | Antenna absolute elevation = ground elevation + technology-specific mast height (Section 5.5) |
| `dist_ant_km` | float | Great-circle distance between the trace point and the antenna |
| `dist_ant_norm` | float | `dist_ant_km` normalized to [0,1] by the dataset's maximum |
| `los_category` | int | 1 = valid line-of-sight link, 0 = tunnel (NLOS-eliminated candidates never appear as rows at all) |
| `path_loss_model` | str | "P1812" (successful computation) or "TUNNEL" |

### 8.3 Raw radio physics

| Column | Type | Description |
|---|---|---|
| `perte_db` | float | ITU-R P.1812 path loss (dB); NaN in tunnels |
| `snr_db` | float | Base SNR before speed/rain adjustment (dB) |
| `snr_adjusted_db` | float | Final SNR: base SNR, minus speed-based degradation, minus rain attenuation, re-capped at the technology ceiling; sentinel in tunnels |
| `snr_norm` | float | `snr_adjusted_db` normalized to [0,1] (Section 6.10) |
| `rtt_ms` | float | Round-trip time from the empirical distance-based model (Section 5.10.1). Weather-invariant — rain only affects the derived `qos_sat_latence` score (Section 6.7.2), not this raw column |
| `rtt_norm` | float | `rtt_ms` normalized to [0,1] (Section 6.10) |
| `ber` | float | Bit error rate, recomputed from the final rain-and-speed-adjusted SNR (Section 6.4.3); floored; worst-case in tunnels |
| `packet_loss_pct` | float | Derived from a coded BER (Section 5.12); recomputed from the rain-adjusted SNR in `enrich_data.py` |
| `delta_vitesse` | float | Orphaned stochastic augmentation term (Section 5.13) — not used in any downstream calculation |
| `delta_snr_db` | float | Orphaned stochastic augmentation term (Section 5.13) — not a weather effect, not used in any downstream calculation |

### 8.4 Throughput

| Column | Type | Description |
|---|---|---|
| `debit_adj_mbps` | float | Final throughput: empirical distance-based estimate, attenuated by rain (Section 6.4.2), plus 2G/3G realism noise (Section 6.5) |
| `debit_adj_p95_mbps` | float | Companion column using the scenario day's peak (p95) hourly rain rate instead of the reference-hour rain rate |
| `debit_adj_norm` | float | `debit_adj_mbps` normalized to [0,1] (Section 6.10) |

### 8.5 Weather

| Column | Type | Description |
|---|---|---|
| `scenario_id` | str | S1 (winter), S2 (spring), S3 (summer), S4 (autumn) |
| `pluie_mm_h` | float | Rain rate at the reference hour, spatially interpolated + perturbed |
| `pluie_p95_mm_h` | float | Peak (worst) hourly rain rate observed on that scenario's day |
| `pluie_norm` | float | Normalized rain rate |
| `atten_pluie_db_km` | float | ITU-R P.838-3 specific rain attenuation (dB/km) for this link's frequency and rain rate |
| `attenuation_pluie_pct` | float | Percent throughput capacity lost to rain, via the Shannon-ratio method |
| `temp_c` | float | Air temperature at the reference hour |
| `temp_norm` | float | Normalized temperature |
| `vent_kmh` | float | Wind speed |
| `humidity_pct` | float | Relative humidity |
| `cloudcover_pct` | float | Cloud cover fraction |

### 8.6 QoS scores and satisfaction components

| Column | Type | Description |
|---|---|---|
| `qos_sat_debit` | float | Throughput satisfaction [0,1] (Section 6.7.1) |
| `qos_sat_latence` | float | Latency satisfaction [0,1], rain-reweighted internally (Section 6.7.2) |
| `qos_sat_ber` | float | BER satisfaction [0,1] (Section 6.7.3) |
| `qos1` | float | Equal-weighted mean of the three satisfaction scores (Section 6.8) |
| `qos2` | float | Throughput-weighted combination of the three satisfaction scores (Section 6.8) |

### 8.7 Cost and infrastructure

| Column | Type | Description |
|---|---|---|
| `cout_installation_eur` | int | One-time install cost (Section 6.6) |
| `cout_mensuel_eur` | int | Recurring monthly cost |
| `cout_norm` | float | Normalized monthly cost |
| `conso_w` | int | Power consumption (W) |
| `tech_class` | int | Reserved for a future non-cellular technology class (e.g. satellite/LPWAN), currently out of scope |
| `tech_label` | str | Same rationale as `tech_class` |

### 8.8 Coverage and quality flags

| Column | Type | Description |
|---|---|---|
| `qualite_ok` | bool | Per-generation binary coverage gate (Section 6.9) — never flips across weather scenarios for the same link in this dataset (a verified, expected property, not a bug) |
| `zone_blanche` | int (0/1) | Per-point coverage gap flag, using a simpler generation-agnostic throughput floor (Section 6.9) |
| `flag_zone_blanche` | int (0/1) | Redundant mirror of `zone_blanche == 1` (Section 6.11) |
| `flag_no_signal` | int (0/1) | Link-level "practically no usable signal" flag (Section 6.11) |
| `flag_debit_nul` | int (0/1) | `debit_adj_mbps == 0` flag (Section 6.11) |

---

## 9. Known Limitations, Assumptions, and Engineering Choices

This section consolidates, in one place, everything a reader should know before treating any number in this dataset as an unquestionable ground truth.

1. Rain attenuation is physically negligible at these frequencies. The dataset correctly shows this (Section 6.4.4) — it is a finding, not a defect. Any downstream model trained on this dataset will not learn a strong "weather changes my decision" behavior, because that behavior is not physically present in the underlying radio channel at these frequencies over these link distances.
2. `qualite_ok` never flips across weather scenarios for a given link, a direct consequence of point 1 combined with `snr_adjusted_db`/`rtt_ms` being otherwise weather-invariant by construction.
3. `qos1` and `qos2` are near-interchangeable on this corridor — the choice between them is a minor modeling decision, not a major one, for this specific dataset.
4. `qos1`/`qos2` are linear combinations of features already present elsewhere in the dataset (Section 6.8) — a circularity that must be avoided in any model that uses both the QoS scores and the raw components as separate inputs to predict a QoS-derived target.
5. All throughput/RTT/BER acceptability thresholds are engineering choices, anchored to the closest applicable standard or regulatory reference, but not literal citations of a norm that mandates that exact number for this exact use case (Sections 6.7, 6.9).
6. `TECH_PROFILES` (Section 5.7) is a synthesized bibliographic table, not a single-source citation — its values represent a good-faith aggregation of typical figures per technology/band, not measured data from this specific corridor.
7. `delta_snr_db` and `delta_vitesse` are inert, orphaned columns (Section 5.13) — present in the CSV, computed with a documented rationale, but not consumed by any other calculation in the pipeline. Do not assume they already encode a physical effect.
8. The mast-height table (Section 5.5) is an engineering assumption about typical deployment patterns, not a citation.
9. The packet size and per-generation coding gains (Section 5.12) are engineering assumptions, order-of-magnitude-consistent with the literature but not tied to a specific standard.
10. Infrastructure costs (Section 6.6) are placeholder engineering figures, not sourced from a published cost study.
11. The vegetation fallback heuristic (Section 5.4) is corridor-specific (calibrated by manual inspection of this exact line) and must not be reused for a different railway line without recalibration.
12. ANFR antenna data is a live registry (Section 3.4) — re-fetching at a different date is not guaranteed to reproduce the exact same antenna set. For strict reproducibility, archive the specific `antennes_anfr.csv` used to generate a given dataset version, rather than relying on re-fetching.
13. The alternative Shannon throughput model exists in the code but was not used to produce the documented dataset (Section 5.10.2) — the empirical model is the one actually used.
14. The stochastic weather fallback exists as a development safety net and was not the path used for the documented dataset, which used real ERA5 data throughout.
15. A single, generic QoS profile is used for evaluation, rather than multiple application-specific profiles (e.g., video call vs. emergency braking vs. infrastructure monitoring). This was a deliberate scope decision, reflecting the project's focus on critical-application requirements for autonomous rail operation, where satisfying the single most stringent requirement along the route is the operationally meaningful target, rather than averaging across multiple simultaneous application profiles. An earlier phase of this work did explore a multi-profile-conditioned approach and found that the optimal antenna choice can depend on the active application profile in a non-trivial fraction of point/scenario combinations — this finding is retained as exploratory analysis but does not underpin the final single-profile architecture documented here, and this scope decision should be stated explicitly, not left implicit, in any report built on this dataset.

---

## 10. Step-by-Step Reproduction Guide

**Prerequisites:**

- Python 3.x with `pandas`, `numpy`, `scipy`, `pyproj`, `geopandas`, `shapely`, `rasterio`, `requests`
- `crc-covlib` (compiled C++ library with Python bindings) — hard dependency, no fallback
- Network access to: ressources.data.sncf.com, Overpass API mirrors, data.anfr.fr, api.opentopodata.org, archive-api.open-meteo.com

**Directory layout expected:**

```
project_root/
├── data/
│   ├── raw/
│   │   └── courpiere_ambert_resampled.csv   (must be provided manually — the initial raw trace)
│   ├── processed/
│   └── cache/
├── fetch_data.py
├── fetch_temperature.py
├── prepare_data.py
└── enrich_data.py
```

**Execution order** (must be run in this exact sequence — each step consumes the previous step's output):

```bash
# Step 1 — static data (trace resampling, tunnels, ANFR antennas, DEM raster)
python fetch_data.py
#   → writes: data/raw/ligne_gps.csv, passages_niveau.csv, tunnels.csv,
#             antennes_anfr.csv, terrain/eu_dem_courpiere_ambert.tif

# Step 1b — weather scenario data (can run in parallel with Step 1, no dependency between them)
python fetch_temperature.py
#   → writes: data/raw/temperature_scenarios.csv

# Step 2 — radio physics (requires Step 1's outputs)
python prepare_data.py
#   optional flags: --step-m 20 --corridor-km 10.0 --vitesse-kmh 80.0 --modele-debit empirique
#   → writes: data/processed/dataset_base.csv

# Step 3 — weather integration, QoS scoring, final assembly (requires Step 1b and Step 2's outputs)
python enrich_data.py --scenario ALL
#   → writes: data/processed/dataset_enrichi.csv   (THE FINAL DATASET)
```

If your intermediate summary numbers diverge meaningfully from a previous reference run, the most likely causes, in order of likelihood, are: (a) a different snapshot of the live ANFR registry (Section 3.4), (b) a different `courpiere_ambert_resampled.csv` starting trace, or (c) a code change relative to the version documented here.

---

## 11. References

- **ITU-R P.1812**: Recommendation ITU-R P.1812, "A path-specific propagation prediction method for point-to-area terrestrial services in the VHF and UHF bands." Implemented via the `crc-covlib` library.
- **ITU-R P.838-3**: Recommendation ITU-R P.838-3, "Specific attenuation model for rain for use in prediction methods."
- **ITU-T Y.1541**: Recommendation ITU-T Y.1541, "Network performance objectives for IP-based services" — latency class framework.
- **3GPP TR 38.913**: "Study on Scenarios and Requirements for Next Generation Access Technologies" — 5G NR throughput/latency targets.
- **3GPP TS 25.101**: UMTS (3G) User Equipment radio transmission and reception specification.
- **ETSI TS 145.001**: GSM/GPRS (2G) physical layer specification.
- **ERA/ERTMS/033281**: European Railway Agency, ERTMS latency requirements reference.
- **ARCEP**: French telecommunications regulator, annual mobile quality-of-service survey methodology (throughput acceptability thresholds).
- Murota, K., & Hirade, K. (1981). "GMSK Modulation for Digital Mobile Radio Telephony." *IEEE Transactions on Communications*, COM-29(7), 1044–1050.
- Proakis, J. G. *Digital Communications*, 5th ed. — standard modulation BER/SNR reference.
- ICAO/WMO International Standard Atmosphere (1975) — adiabatic lapse rate, used for weather altitude correction.
- ESA WorldCover — global 10 m land cover classification, used for vegetation/clutter categorization.
- Copernicus EU-DEM 25 m — digital elevation model, via OpenTopoData API and local raster construction.
- ERA5 reanalysis (Copernicus Climate Change Service), accessed via the Open-Meteo Archive API — hourly historical weather data source.
- ANFR (Agence Nationale des Fréquences) — French national mobile antenna registry, `observatoire_2g_3g_4g` dataset.
- SNCF Open Data — railway infrastructure reference data (line geometry, level crossings).
