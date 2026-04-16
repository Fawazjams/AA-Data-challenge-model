# NavRisk — Airline Crew Sequence Risk System

**EPPS x American Airlines Data Analytics Challenge — GROW 26.2**

NavRisk identifies sub-optimal pilot crew sequences for American Airlines' DFW hub. Given an inbound flight (A → DFW) and an outbound flight (DFW → B), the system predicts whether that pair should be avoided in pilot scheduling to minimize delay propagation, duty time violations, missed connections, and weather risk.

**Live Demo:** [aa-data-challenge-model.vercel.app](https://aa-data-challenge-model.vercel.app)

---

## How It Works

```
BTS Flight Data (6M+ records)
        ↓
Feature Engineering (27 features across 4 risk categories)
        ↓
XGBoost Classifier (5-fold stratified CV, 0.89 AUC-ROC)
        ↓
36,052 scored airport pairs × 4 seasons
        ↓
Interactive web app with live AWC weather overlay
```

## Project Structure

```
├── notebooks/
│   ├── analysis.py          # Full ML pipeline — data → features → XGBoost → export
│   ├── export_json.py       # Convert model outputs to frontend JSON
│   └── outputs/             # Generated charts and CSVs
├── frontend/                # Next.js 16 web application
│   ├── src/app/
│   │   ├── page.tsx         # Interactive flight map with animated paths
│   │   ├── model/page.tsx   # Model rundown — features, validation, data sources
│   │   └── api/weather/     # Server-side proxy for AWC METAR API
│   ├── src/lib/data.ts      # Data types, risk helpers, live weather logic
│   └── public/data/         # Static JSON files consumed by the frontend
├── data/                    # Raw CSVs (not committed — see setup below)
├── NavRisk_Team_Report.md   # Full team report (~20 pages)
└── README.md
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Pipeline | Python 3, pandas, NumPy, XGBoost, scikit-learn, NetworkX |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| Map | react-simple-maps, d3-geo |
| Live Weather | Aviation Weather Center (AWC) METAR API |
| Deployment | Vercel |

## Model Performance

| Metric | Score |
|--------|-------|
| Recall (5-fold CV) | 0.89 |
| Precision | 0.85 |
| F1 Score | 0.87 |
| AUC-ROC | 0.89 |
| Temporal Holdout Recall | 0.86 |
| Propagation Catch Rate | 91% |

## Data Sources

- **BTS On-Time Performance** (primary) — cause-decomposed delay data per airline per airport
- **AWC METAR** (live integrated) — real-time flight category, wind, visibility, ceiling
- **Airport Geospatial Data** — coordinates and US regions for geographic risk features
- FAA ASPM, NOAA, OpenSky Network — recommended for production enhancement

## Setup

### Prerequisites

- Python 3.9+
- Node.js 18+

### 1. Data

Download `Airline_Delay_Cause.csv` from [BTS](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) and `airport_codes.csv` from [GitHub](https://github.com/datasets/airport-codes). Place both in a `data/` folder at the project root.

### 2. Run the ML pipeline

```bash
cd notebooks
pip install pandas numpy xgboost scikit-learn networkx matplotlib
python analysis.py
python export_json.py
```

This trains the model and exports JSON files to `frontend/public/data/`.

### 3. Run the web app

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

Open [localhost:3000](http://localhost:3000).

## Features

- **Interactive US map** — click airports to build crew sequences with animated flight paths
- **Season selector** — risk profiles change dramatically between Spring, Summer, Fall, Winter
- **Live weather overlay** — AWC METAR data adjusts risk scores in real-time
- **Risk severity matrix** — shows which factor (propagation, duty, turnaround, weather) drives each pair
- **Model rundown page** — full pipeline explanation, feature importance, validation metrics
- **Data source documentation** — what's used, integrated, and recommended for production

---

*Built for the EPPS x American Airlines Data Analytics Challenge — GROW 26.2*
