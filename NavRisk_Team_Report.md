# NavRisk: Identifying Sub-Optimal Pilot Crew Sequences Using Machine Learning

**EPPS × American Airlines Data Analytics Challenge — GROW 26.2**

**Live Demo:** [navrisk.vercel.app](https://aa-data-challenge-model.vercel.app)  
**Source Code:** [github.com/ArpitKhavate/AA-Data-challenge-model](https://github.com/ArpitKhavate/AA-Data-challenge-model)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Architecture](#3-solution-architecture)
4. [Data Sources](#4-data-sources)
5. [Data Ingestion & Preprocessing](#5-data-ingestion--preprocessing)
6. [Feature Engineering](#6-feature-engineering)
7. [Airport Pair Construction](#7-airport-pair-construction)
8. [Model Selection: Why XGBoost](#8-model-selection-why-xgboost)
9. [Model Training & Configuration](#9-model-training--configuration)
10. [Handling Seasonality](#10-handling-seasonality)
11. [Handling Sparse Severe Weather Events](#11-handling-sparse-severe-weather-events)
12. [Model Validation & Metrics](#12-model-validation--metrics)
13. [Out-of-Fold Predictions & Confidence Calibration](#13-out-of-fold-predictions--confidence-calibration)
14. [Live Weather Integration](#14-live-weather-integration)
15. [Network Graph Analysis](#15-network-graph-analysis)
16. [Interactive Web Application](#16-interactive-web-application)
17. [Key Findings](#17-key-findings)
18. [Limitations](#18-limitations)
19. [Future Work & Production Roadmap](#19-future-work--production-roadmap)
20. [Conclusion](#20-conclusion)

---

## 1. Executive Summary

NavRisk is an end-to-end machine learning system that identifies sub-optimal pilot crew sequences for American Airlines' DFW hub network. Given the challenge of determining which inbound–outbound flight pairs (A → DFW → B) should not be assigned to the same pilot, we built an XGBoost classification model trained on 6M+ BTS flight records across 141 airports and 4 seasons.

The system evaluates **36,052 unique airport pair × season combinations** across four risk dimensions:

- **Delay Propagation** — cascading delays from late-arriving aircraft
- **Duty Time Violations** — accumulated delays pushing pilots toward FAA 14-hour limits
- **Missed Connections** — turnaround window failures at DFW
- **Operational & Weather Risk** — systemic disruptions from weather, ATC, and correlated regional storms

The model achieves **89% recall** (5-fold stratified CV), **0.89 AUC-ROC**, and a **91% propagation catch rate** — meaning it correctly identifies 91% of double-cascade pairs. An interactive web application with live AWC METAR weather integration allows real-time risk assessment of any sequence.

---

## 2. Problem Statement

Airlines schedule pilots into "sequences" — chains of flights a pilot operates over several days. When both the inbound flight (A → DFW) and outbound flight (DFW → B) are prone to delays, the sequence becomes a bottleneck:

1. A delayed inbound flight compresses the turnaround window at DFW
2. A compressed turnaround leads to a late outbound departure
3. The delay **propagates** — the pilot's entire sequence shifts, risking duty time violations
4. If both airports share weather vulnerabilities (e.g., both in the Southeast during hurricane season), the risk compounds

**Objective:** Identify pairs of airports (A and B) that represent sub-optimal combinations so they can be avoided in future crew sequence assignments.

---

## 3. Solution Architecture

Our solution consists of three components:

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   Python ML Pipeline │────▶│   JSON Data Export    │────▶│  Next.js Web App    │
│   (analysis.py)      │     │   (8 static files)    │     │  (Interactive Map)  │
│                      │     │                       │     │                     │
│  • Data cleaning     │     │  • all_pairs.json     │     │  • Flight Map page  │
│  • Feature eng.      │     │  • top_pairs.json     │     │  • Model Rundown    │
│  • XGBoost training  │     │  • season_stats.json  │     │  • Live METAR API   │
│  • OOF predictions   │     │  • feature_imp.json   │     │  • Risk calculator  │
│  • Network graph     │     │  • model_metrics.json │     │  • Animated paths   │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

**Tech Stack:**
- **Backend/ML:** Python 3, pandas, NumPy, XGBoost, scikit-learn, NetworkX, matplotlib
- **Frontend:** Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4, react-simple-maps, d3-geo
- **Live Data:** Aviation Weather Center (AWC) METAR API via server-side proxy
- **Deployment:** Vercel (auto-deploy from GitHub)

---

## 4. Data Sources

### 4.1 Primary: BTS On-Time Performance (Used)

**Source:** Bureau of Transportation Statistics — [transtats.bts.gov](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)

This is the only public dataset that provides **cause-decomposed delay minutes per airline per airport per month**. We use the `Airline_Delay_Cause.csv` dataset, which contains:

| Column | Purpose |
|--------|---------|
| `late_aircraft_delay` | **Direct measure of delay propagation** — how often a late-arriving aircraft caused the next departure to be late |
| `late_aircraft_ct` | Frequency of propagation events |
| `weather_delay` / `weather_ct` | Weather-caused delays and counts |
| `carrier_delay` / `carrier_ct` | Mechanical/crew-caused delays |
| `nas_delay` / `nas_ct` | ATC and National Airspace System delays |
| `arr_del15` | Flights arriving 15+ minutes late (turnaround risk) |
| `arr_cancelled` / `arr_diverted` | Cancellation and diversion rates |
| `arr_flights` | Total flights (denominator for all rates) |

**Why BTS as primary:** It is the only source that separates delay causes into the exact categories we need — propagation, weather, carrier, and NAS — at the airline level. This lets us build features that directly map to the four competition objectives.

**Coverage:** 6M+ American Airlines flight records, 141 airports, years 2003–2024, monthly granularity.

### 4.2 Integrated: AWC METAR (Live)

**Source:** Aviation Weather Center — [aviationweather.gov](https://aviationweather.gov)

Real-time METAR (Meteorological Aerodrome Report) observations are fetched via a server-side API proxy to inject **current conditions** into the risk assessment. Each METAR provides:

- **Flight category** (VFR / MVFR / IFR / LIFR) — directly correlates with delay probability
- **Wind speed and gusts** — high gusts cause ground stops
- **Visibility** — low visibility triggers instrument approaches and spacing increases
- **Ceiling height** — determines the approach procedure required
- **Active weather phenomena** (thunderstorms, freezing rain, etc.)

When live conditions are degraded, a weather multiplier boosts the historical risk score:

| Flight Category | Multiplier |
|-----------------|------------|
| VFR | 1.0× |
| MVFR | 1.15× |
| IFR | 1.35× |
| LIFR | 1.50× |
| + Thunderstorms | +0.20× |
| + Gusts > 35kt | +0.10× |

### 4.3 Used: Airport Geospatial Data

**Source:** [github.com/datasets/airport-codes](https://github.com/datasets/airport-codes)

IATA/ICAO codes, geographic coordinates, state, and US region for each airport. Enables the `same_region` feature — airports sharing regional weather systems (e.g., Southeast hurricane belt, Midwest tornado alley) carry compounded risk when paired.

### 4.4 Recommended for Production

| Source | What It Would Add |
|--------|-------------------|
| **FAA ASPM** (aspm.faa.gov) | Direct airport efficiency scores, taxi times, gate-to-gate delay data |
| **NOAA Historical Weather** (weather.gov) | Hourly precipitation type, wind gusts, and visibility per airport |
| **OpenSky Network** (opensky-network.org) | Real-time ADS-B traffic density as a congestion proxy |
| **OpenWeather API** (openweathermap.org) | Forecast data to predict risk 24–72 hours ahead |

---

## 5. Data Ingestion & Preprocessing

### Step 1: Load and Filter

We load the BTS delay dataset and filter to American Airlines only (`carrier == 'AA'`), producing ~23,000 rows (airport × month × year).

### Step 2: Numeric Conversion

All delay columns are converted to numeric types. Missing values are filled with 0 (no delay observed = safe).

### Step 3: Safe Denominators

To avoid division-by-zero when computing delay rates, we replace any `arr_flights == 0` with 1:

```python
safe_flights = aa['arr_flights'].replace(0, 1)
```

### Step 4: Season Assignment

Each month is mapped to a season label:

| Months | Season |
|--------|--------|
| March, April, May | Spring |
| June, July, August | Summer |
| September, October, November | Fall |
| December, January, February | Winter |

This is critical because the same airport has vastly different risk profiles across seasons.

---

## 6. Feature Engineering

We create **10 airport-level features** organized into four risk categories matching the competition objectives:

### 6.1 Propagation Risk Features

| Feature | Formula | What It Captures |
|---------|---------|-----------------|
| `propagation_risk` | `late_aircraft_delay / flights` | Average minutes of cascading delay per flight |
| `propagation_freq` | `late_aircraft_ct / flights` | How often (not just how much) delays cascade |

**Why these matter:** `late_aircraft_delay` is the single most important column for this problem. It directly measures what the competition calls "delay propagation" — a late-arriving aircraft causing the next departure to be late. In a pilot sequence, that next departure IS the DFW → B leg.

### 6.2 Duty Time Risk Features

| Feature | Formula | What It Captures |
|---------|---------|-----------------|
| `duty_burden` | `(weather + carrier + NAS + late_aircraft) / flights` | Total delay minutes per flight from all sources |
| `carrier_risk` | `carrier_delay / flights` | Mechanical and crew-caused delays eating into duty hours |

**Why these matter:** The FAA limits pilots to 14 hours of duty time. Every minute of delay at either airport accumulates against this limit. High duty burden means the pilot is likely to approach legal limits, requiring re-routing or rest breaks.

### 6.3 Turnaround Risk Features

| Feature | Formula | What It Captures |
|---------|---------|-----------------|
| `turnaround_risk` | `arr_del15 / flights` | Probability of 15+ minute arrival delay |
| `cancel_rate` | `arr_cancelled / flights` | Cancellation probability |
| `divert_rate` | `arr_diverted / flights` | Diversion probability |

**Why these matter:** DFW turnaround windows can be as short as 45 minutes. A 15+ minute arrival delay destroys that buffer. Cancellations and diversions are the worst-case turnaround outcomes.

### 6.4 Weather & Systemic Risk Features

| Feature | Formula | What It Captures |
|---------|---------|-----------------|
| `weather_risk` | `weather_delay / flights` | Average weather delay per flight |
| `weather_freq` | `weather_ct / flights` | How often weather causes delays |
| `nas_risk` | `nas_delay / flights` | NAS/ATC delays, often caused by weather elsewhere |

### 6.5 Aggregation

All features are computed per airport per season, averaged across all available years. This creates a **stable risk profile** for each airport in each season, smoothing out year-to-year noise while preserving seasonal patterns.

---

## 7. Airport Pair Construction

### Cross-Join

For each season, we create every possible (A, B) airport pair via a cross-join, producing ~9,000 pairs per season × 4 seasons = ~36,000 total pairs.

### Pair-Level Combined Features

Beyond individual airport features, we create **interaction features** that capture the combined effect of pairing two airports:

| Feature | Formula | Interpretation |
|---------|---------|---------------|
| `combined_propagation` | `prop_risk_A + prop_risk_B` | Total cascading delay burden across both legs |
| `both_propagation_prone` | 1 if both above median | "Double cascade" flag — highest danger |
| `combined_duty_burden` | `duty_A + duty_B` | Total duty hour exposure across the sequence |
| `max_turnaround_risk` | `max(turn_A, turn_B)` | Worst-leg determines sequence reliability |
| `combined_cancel_risk` | `cancel_A + cancel_B` | Total cancellation exposure |
| `both_weather_prone` | 1 if both above median | Both airports weather-vulnerable |
| `same_region` | 1 if same US region | Shared storm systems (hurricane belt, tornado alley) |
| `season_num` | 0–3 encoding | Allows model to learn seasonal weight directly |

### Label Creation

We define **high risk** as the top 30% of pairs by composite risk score:

```python
composite_risk = combined_duty_burden × season_weight
threshold = pairs['composite_risk'].quantile(0.70)
high_risk = (composite_risk >= threshold)
```

Season weights encode expert knowledge about relative seasonal danger:
- Spring: 1.3× (thunderstorm season)
- Summer: 1.2× (peak traffic)
- Winter: 1.1× (ice/snow cascades)
- Fall: 0.9× (calmest operationally)

---

## 8. Model Selection: Why XGBoost

We chose XGBoost (Extreme Gradient Boosting) for several reasons specific to this problem:

**Non-linear interactions:** High propagation at airport A combined with high weather at airport B is MORE dangerous than either alone. A linear model cannot learn this multiplicative interaction. XGBoost can — its decision trees naturally capture feature interactions.

**Handles mixed feature types:** Our features span continuous values (delay minutes), binary flags (both_propagation_prone), and categorical encodings (season). XGBoost handles all of these natively.

**Built-in class imbalance handling:** The `scale_pos_weight` parameter compensates for having fewer high-risk pairs than low-risk pairs, which is essential for rare severe weather events.

**Interpretability:** Feature importance scores directly tell us which risk categories the model relies on — critical for explaining results to domain experts and judges.

**Robustness to noise:** Flight delay data is inherently noisy. XGBoost's ensemble of 100 trees averages out individual prediction errors.

---

## 9. Model Training & Configuration

### Hyperparameters

```python
XGBClassifier(
    n_estimators=100,    # 100 boosted trees
    max_depth=4,         # Shallow trees prevent overfitting
    learning_rate=0.1,   # Conservative learning rate
    scale_pos_weight=ratio,  # Class imbalance correction
    eval_metric='logloss',
    random_state=42
)
```

### Leakage Prevention

The label is derived from `composite_risk = combined_duty_burden × season_weight`. To prevent data leakage, we **remove both `combined_duty_burden` and `season_weight`** from the model's input features. The model must learn risk from the underlying component features, not from the formula used to create the label.

After removing leakage-prone features, the model trains on **27 features**.

### Class Imbalance

High-risk pairs represent approximately 30% of the dataset. We use `scale_pos_weight = negative_count / positive_count` to upweight the minority class, ensuring the model doesn't learn to predict "low risk" for everything.

---

## 10. Handling Seasonality

**Question:** "How would you deal with the seasonality of the data?"

Our approach treats seasonality as a **first-class dimension**, not a correction:

1. **All features are computed per-season.** ORD in April (Spring) and ORD in October (Fall) have completely separate risk profiles. The model sees them as different data points.

2. **Season-specific weights** encode domain knowledge about which seasons are operationally worse:
   - Spring (1.3×): Peak thunderstorm season, highest `weather_delay` rates
   - Summer (1.2×): Peak traffic volume, highest `nas_delay` rates
   - Winter (1.1×): Ice/snow creates the most severe propagation cascades
   - Fall (0.9×): Typically the calmest season

3. **The `season_num` feature** allows XGBoost to learn its own seasonal adjustments from the data, beyond our expert-assigned weights.

4. **Season-stratified validation** ensures every cross-validation fold contains all four seasons proportionally.

**Result:** The model shows dramatically different risk distributions by season:

| Season | Avg Risk | High-Risk % |
|--------|----------|-------------|
| Summer | 79.6% | 78.0% |
| Spring | 30.6% | 29.6% |
| Winter | 10.4% | 9.5% |
| Fall | 0.8% | 0.4% |

This matches aviation domain knowledge: summer thunderstorms and spring storms drive the most disruptions.

---

## 11. Handling Sparse Severe Weather Events

**Question:** "What issues might arise from the sparsity of severe weather events?"

Severe weather events (ice storms, derechos, hurricane landfalls) are rare but devastating. They create three problems for machine learning:

### Problem 1: Small Sample Size

**Solution: Minimum Flight Threshold.** Airports with fewer than 100 total AA flights across all years are excluded. With sparse data, a single severe weather event can dominate the risk profile, creating misleading signals indistinguishable from noise.

```python
MIN_FLIGHTS = 100
risk_profile = risk_profile[risk_profile['total_flights'] >= MIN_FLIGHTS]
```

### Problem 2: Class Imbalance

**Solution: `scale_pos_weight`.** Without correction, the model learns to predict "low risk" for everything since most pairs are safe. XGBoost's built-in class weighting upweights the minority (high-risk) class:

```python
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
```

### Problem 3: Historical Data Can't Capture Current Storms

**Solution: Real-time weather overlay.** Live METAR data from AWC creates a complementary layer. When IFR conditions, thunderstorms, or high gusts are active at either airport, a weather multiplier boosts the historical risk score — catching conditions the training data statistically underrepresents.

```python
adjusted_risk = min(historical_risk × weather_multiplier, 1.0)
```

---

## 12. Model Validation & Metrics

**Question:** "What accuracy metrics would be appropriate?"

We use **recall as the primary metric** because missing a truly dangerous pair (false negative) is far worse than occasionally flagging a safe one (false positive). An airline would rather over-warn than under-warn.

### 12.1 Stratified 5-Fold Cross-Validation

Each fold preserves both the class distribution and season balance.

| Metric | Score |
|--------|-------|
| **Recall** | **0.89** |
| Precision | 0.85 |
| F1 Score | 0.87 |

### 12.2 Temporal Holdout Validation

Train on data from years **before 2020**, test on data from years **2020 and after**. This simulates real deployment where the model predicts future risk from past patterns.

| Metric | Score |
|--------|-------|
| **Recall** | **0.86** |
| Precision | 0.82 |
| F1 Score | 0.84 |

The temporal holdout is a stricter test than random folds — it verifies the model generalizes to future time periods, not just held-out random samples.

### 12.3 Additional Metrics

| Metric | Score | Meaning |
|--------|-------|---------|
| **AUC-ROC** | **0.89** | Strong discrimination between high and low-risk pairs |
| **Propagation Catch Rate** | **91%** | 91% of "double-cascade" pairs (both airports above median propagation) are correctly flagged |

### 12.4 Why These Metrics

- **Recall (primary):** A missed dangerous pair means a pilot gets assigned an unsafe sequence. This is operationally unacceptable.
- **Precision:** An over-flagged pair means the scheduler has fewer options. This is inconvenient but not dangerous.
- **AUC-ROC:** Measures overall ranking quality — can the model correctly sort pairs from safest to most dangerous?
- **Propagation Catch Rate:** A domain-specific metric — the competition specifically asks about delay propagation. This measures how well we catch the worst cascading-delay pairs.

---

## 13. Out-of-Fold Predictions & Confidence Calibration

### The Problem with In-Sample Prediction

If we train on all data and predict the same data, XGBoost memorizes many pairs and produces near-certain probabilities (95–100% confidence). This is misleading.

### Solution: Out-of-Fold Predictions

We use 5-fold stratified cross-validation where each pair's risk score comes from a model that **never saw that pair during training**:

```python
oof_probs = np.zeros(len(X))
for train_idx, test_idx in StratifiedKFold(5).split(X, y):
    fold_model = XGBClassifier(...)
    fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_probs[test_idx] = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
```

These "honest" probabilities are what we export and display on the web application.

### Confidence Calibration

Raw OOF probabilities cluster near 0 and 1. For human interpretability, we apply a sigmoid compression so most pairs land in the 55–90% confidence band:

```
calibrated = 0.5 + 0.45 × tanh(2.5 × (raw - 0.65))
```

---

## 14. Live Weather Integration

### Architecture

The web application fetches live METAR data from AWC via a Next.js API route (server-side proxy to bypass CORS):

```
User clicks airport → Frontend → /api/weather/metar?ids=KORD,KMIA → AWC API → Response
```

### Risk Adjustment

When both airports are selected, the live METAR data is analyzed:

1. **Flight category** determines the base multiplier (VFR=1.0× through LIFR=1.5×)
2. **Thunderstorms** in the METAR raw string add 0.2× to the multiplier
3. **High gusts** (>35 knots) add 0.1×
4. The final adjusted risk = `min(historical_risk × multiplier, 1.0)`

This bridges the gap between historical patterns (what BTS tells us about typical behavior) and current conditions (what AWC tells us about right now).

---

## 15. Network Graph Analysis

All 36,052 scored pairs form a network graph where:
- **Nodes** = 141 airports
- **Edges** = airport pairs, weighted by average risk across seasons
- **Red edges** (≥ 0.7 risk) = high-risk pairs to avoid
- **Green edges** (< 0.7 risk) = acceptable pairs

**Network statistics:**
- 141 nodes (airports with sufficient data)
- 9,848 edges (unique pairs)
- 1,210 edges flagged as high-risk

The graph enables schedulers to query any sequence instantly: given airports A and B, the edge weight is the risk score.

---

## 16. Interactive Web Application

The NavRisk web application provides two pages:

### Flight Map (Main Page)

- **Interactive US map** with all 141 airports as clickable dots
- **Season selector** to switch between Spring, Summer, Fall, Winter
- **Airport selection** via map click or dropdown — triggers animated curved flight path from A → DFW (blue) and DFW → B (red) with a traveling airplane icon
- **Real-time risk assessment** showing risk probability, confidence, duty violation probability, and turnaround risk
- **Live METAR conditions** for both selected airports with flight category badges
- **Live-adjusted risk** showing the weather-multiplied score when conditions are degraded
- **Risk severity matrix** breaking down the contribution of each risk factor
- **Delay propagation explanation** with a visual cascade diagram

### Model Rundown Page

- **Decision flow diagram** showing the 6-step pipeline from data ingestion to actionable output
- **Feature importance chart** with color-coded risk categories
- **Top 10 flagged pairs** with primary risk driver identification
- **Seasonal risk summary** showing how risk distributes across seasons
- **Data sources section** documenting all sources with integration status
- **Sparse weather handling** section explaining our three-part approach
- **Model validation metrics** with stratified CV and temporal holdout results
- **Limitations and future work** for honest constraint documentation

---

## 17. Key Findings

### 17.1 Feature Importance

The model's top features reveal what drives crew sequence risk:

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | `max_turnaround_risk` | 0.710 | Turnaround |
| 2 | `combined_propagation` | 0.082 | Propagation |
| 3 | `both_propagation_prone` | 0.031 | Propagation |
| 4 | `season_num` | 0.020 | Seasonal |
| 5 | `duty_A` | 0.018 | Duty Time |

**Insight:** Turnaround risk dominates — the probability of a 15+ minute delay at either airport is the strongest predictor of sequence failure. This makes operational sense: a tight DFW turnaround window is the critical bottleneck.

### 17.2 Seasonal Patterns

- **Summer is the most dangerous season** — 78% of all pairs are flagged as high-risk
- **Fall is the safest** — only 0.4% high-risk
- **Spring** shows elevated risk from thunderstorms (29.6% high-risk)
- **Winter** shows moderate risk from ice/snow cascades (9.5% high-risk)

### 17.3 Actionable Recommendations

The top flagged pairs (all Summer season) should be **removed from pilot crew sequences** or given additional turnaround buffer time. These pairs consistently show:
- Both airports above median propagation risk
- Combined duty burden exceeding safe thresholds
- Historical turnaround failure rates above 30%

---

## 18. Limitations

We acknowledge these constraints honestly:

1. **Labels are derived from composite risk** (a statistical proxy), not ground-truth disruption outcomes from actual crew operations.
2. **No actual crew schedule data** — duty violation probability is estimated from delay accumulation rather than real FAA duty time tracking.
3. **BTS `weather_delay` is an aggregated proxy** — actual hourly weather conditions (METAR/TAF) would provide finer-grained risk signals.
4. **Historical patterns assume future similarity** — significant route network changes or new weather patterns could shift risk profiles.
5. **Model evaluates 2-leg sequences** (A → DFW → B) only. Real pilot sequences span multiple days and many legs.

---

## 19. Future Work & Production Roadmap

1. **Integrate actual crew scheduling constraints** — FAA rest rules, union agreements, aircraft-specific turnaround minimums
2. **Incorporate NOAA weather reanalysis data** — hourly precipitation, wind, and visibility per airport for the training data
3. **Deploy as a real-time API** that crew scheduling systems query before assigning a pilot to a sequence
4. **Add a feedback loop** from actual disruption outcomes (delayed sequences, duty violations) to continuously retrain the model
5. **Extend to multi-leg sequences** using graph-based optimization (shortest path through low-risk edges)
6. **SMOTE oversampling** of rare severe-weather pairs to further address sparsity
7. **Ensemble methods** combining gradient boosting with weather-specific models

---

## 20. Conclusion

NavRisk demonstrates a complete, deployable approach to identifying sub-optimal pilot crew sequences. By combining BTS historical delay data with real-time AWC weather observations and an XGBoost classification model, we provide:

- **Quantified risk scores** for every airport pair × season combination
- **Explainable results** showing exactly which risk factors drive each pair's score
- **Actionable recommendations** that crew schedulers can use immediately
- **Live weather awareness** that adjusts risk in real-time
- **An interactive tool** for exploring and validating the model's outputs

The system identifies 1,210 high-risk airport pair connections that American Airlines should consider avoiding in pilot crew sequences, with summer operations requiring the most careful scheduling attention.

---

*Built with XGBoost · Next.js · AWC METAR · react-simple-maps*  
*EPPS × American Airlines Data Analytics Challenge — GROW 26.2*
