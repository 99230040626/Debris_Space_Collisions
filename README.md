# AI-Powered Space Debris Collision Prediction System

An AI-based system that analyzes satellite and space debris orbital data to predict potential collision risks and visualize orbital trajectories using machine learning and simulation techniques.

---

## Project Overview

This project implements a complete pipeline for space debris collision risk assessment:

| Stage | Module | Description |
|-------|--------|-------------|
| **Data Loading** | `src/data_loader.py` | Parses TLE (Two-Line Element) files |
| **Preprocessing** | `src/preprocessor.py` | Derives orbital parameters; builds feature matrices |
| **Trajectory Simulation** | `src/trajectory_simulator.py` | SGP4 orbit propagation |
| **Collision Detection** | `src/collision_detector.py` | Conjunction (close-approach) event detection |
| **ML Risk Estimation** | `src/ml_model.py` | Random Forest collision probability classifier |
| **Visualization** | `src/visualizer.py` | Matplotlib & Plotly orbital charts |
| **Dashboard** | `dashboard/app.py` | Streamlit interactive web dashboard |

---

## Project Structure

```
Debris_Space_Collisions/
├── data/
│   └── sample_tle_data.txt      # Example TLE dataset (20 objects)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # TLE parsing and loading
│   ├── preprocessor.py          # Orbital parameter derivation & feature engineering
│   ├── trajectory_simulator.py  # SGP4-based trajectory propagation
│   ├── collision_detector.py    # Close-approach (conjunction) detection
│   ├── ml_model.py              # Random Forest collision probability model
│   └── visualizer.py            # Matplotlib & Plotly visualization helpers
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── tests/
│   └── test_modules.py          # Unit tests (43 tests, pytest)
├── main.py                      # CLI entry-point
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or later
- `pip`

### 2. Clone & install

```bash
git clone <repo-url>
cd Debris_Space_Collisions
pip install -r requirements.txt
```

### 3. Run the CLI pipeline

```bash
python main.py
```

Optional flags:

```
--tle PATH           Path to TLE data file (default: data/sample_tle_data.txt)
--duration MINUTES   Simulation duration in minutes (default: 90)
--step SECONDS       Time step in seconds (default: 60)
--threshold KM       Close-approach distance threshold in km (default: 5)
--min-alt KM         Minimum altitude filter in km (default: 200)
--max-alt KM         Maximum altitude filter in km (default: 2000)
--no-filter          Disable altitude filtering
--plot               Show 3-D trajectory plot (requires a display)
```

Example:

```bash
python main.py --duration 180 --threshold 10 --plot
```

### 4. Launch the interactive dashboard

```bash
streamlit run dashboard/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

Use the sidebar to configure simulation parameters and click **Run Simulation**.

### 5. Run tests

```bash
pytest tests/test_modules.py -v
```

---

## Technologies

| Library | Purpose |
|---------|---------|
| **NumPy** | Numerical computations |
| **Pandas** | Tabular data manipulation |
| **scikit-learn** | Random Forest ML model, preprocessing |
| **sgp4** | SGP4 orbital mechanics (TLE propagation) |
| **Matplotlib** | Static 3-D orbit plots |
| **Plotly** | Interactive 3-D orbit charts |
| **Streamlit** | Web dashboard |
| **SciPy** | Scientific utilities |

---

## Data Format

The system uses **TLE (Two-Line Element)** sets — the standard format used by NORAD and space agencies to describe satellite orbits.  A sample dataset of 20 objects (active satellites + debris) is provided in `data/sample_tle_data.txt`.

Real-time TLE data is freely available from [CelesTrak](https://celestrak.org) and [Space-Track](https://www.space-track.org).

---

## Machine Learning Model

A **Random Forest** classifier estimates collision probability from the following features:

- Miss distance (km)
- Relative velocity (km/s)
- Orbital inclination of each object
- Eccentricity of each object
- Mean motion of each object
- BSTAR drag coefficient of each object

Risk levels are categorised as: **LOW** / **MEDIUM** / **HIGH** / **CRITICAL**.

> **Note:** The model is trained on synthetically generated conjunction data for demonstration purposes. Replace with real conjunction screening data (e.g. from Space-Track CDMs) for production use.

---

## License

See [LICENSE](LICENSE).
