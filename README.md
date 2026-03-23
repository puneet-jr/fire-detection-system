# Hybrid AI Fire Detection

This project is a software-first, embedded-systems-friendly fire detection stack built around the sensors you actually have:

- temperature
- humidity
- flame sensor
- buzzer output

Instead of using only fixed thresholds, the project combines:

- supervised machine learning for `normal / warning / fire`
- temporal feature engineering
- anomaly scoring inspired by your Arduino Z-score logic
- real-time simulation
- a Flask API for IoT-style deployment

## Project Structure

- `generate_dataset.py` creates a larger synthetic dataset for laptop testing
- `train_model.py` trains and saves the hybrid ML model bundle
- `simulation.py` runs a live software-only demo
- `server.py` exposes a prediction API
- `visualize.py` saves plots for your report or PPT
- `src/fire_ai/` contains the real implementation

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

> Dataset location override: set `FIRE_DATASET_ROOT` to your folder with `carton_*.csv`, `clothing_*.csv`, `electrical_*.csv`, `Algerian_forest_fires_dataset_CLEANED.csv`, and `forestfires.csv`. Example:

```bash
set FIRE_DATASET_ROOT=C:\Users\punee\Desktop\EMBEDDED\PROJECT\datasets_used  # Windows PowerShell/CMD
export FIRE_DATASET_ROOT=/path/to/datasets_used                               # macOS/Linux
```

Generate the unified dataset from your real files:

```bash
python generate_dataset.py --dataset-root "C:\Users\punee\Desktop\EMBEDDED\PROJECT\datasets_used"
```

Train the model:

```bash
python train_model.py --dataset-root "C:\Users\punee\Desktop\EMBEDDED\PROJECT\datasets_used" --regenerate-dataset
```

Run the real-time simulation:

```bash
python simulation.py
```

Run the API:

```bash
python server.py
```

Create plots:

```bash
python visualize.py
```

## Demo & Test Commands (for viva)

- Quick dataset sanity check: `python generate_dataset.py --output data/fire_sensor_dataset.csv --no-synthetic`
- Train fresh: `python train_model.py --regenerate-dataset --synthetic-sequences 160`
- Inspect metrics: `type models\training_metrics.json` (Windows) or `cat models/training_metrics.json`
- Run live simulation log: `python simulation.py --steps 40 --sleep 0.6`
- Start API server: `python server.py`
- Test API with one reading (normal):

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" ^
	-d "{\"temperature\": 30.5, \"humidity\": 55.0, \"flame\": 0}"
```

- Test API with a fire scenario:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" ^
	-d "{\"temperature\": 55.0, \"humidity\": 20.0, \"flame\": 1}"
```

## What The ML System Implements

The training pipeline creates temporal and risk-aware features from raw sensor values:

- `temp_rate`: how fast temperature is rising
- `humidity_rate`: how fast humidity is falling
- `temp_zscore`: anomaly strength relative to recent baseline
- `humidity_zscore`: anomaly strength for humidity
- `rolling_temp` and `rolling_humidity`: short-term temporal smoothing
- `dryness_index`: high temperature combined with low humidity
- `flame_persistence`: whether flame sensor remains active
- `rule_score`: research-style feature fusion from sensor evidence

Then it trains:

- a multiclass classifier for `normal`, `warning`, and `fire`
- an `IsolationForest` anomaly detector on normal conditions only

At inference time, both are fused. This gives you a hybrid AI system:

- statistical anomaly detection
- supervised ML classification
- rule-based safety override

## How Your Arduino Code Maps To This Project

Your Arduino sketch already implements a good baseline safety layer:

- it calibrates baseline temperature and humidity over 50 samples
- it computes Z-scores for temperature and humidity
- it triggers the buzzer if flame is detected
- it also triggers on abnormal temperature rise or temperature-rise-plus-humidity-drop

That means your current code is already doing:

- baseline modeling
- anomaly detection
- sensor fusion
- alarm logic

This project upgrades that idea by:

- keeping the anomaly concept
- converting the live readings into richer ML features
- learning a classifier from labeled data
- supporting laptop simulation before hardware deployment

## How To Use With Hardware Later

For your laptop demo, use `simulation.py`.

For hardware later, keep your Arduino for sensor reading and buzzer actuation, and send serial data to a Python inference script or API in this format:

```text
temperature=34.5,humidity=51.0,flame=0
```

The same model logic can then return:

- predicted state
- confidence
- anomaly flag
- buzzer recommendation

## Suggested Viva Description

You can say:

> I implemented a hybrid AI-based fire detection system for embedded sensing. The system uses temperature, humidity, and flame sensor data, then derives temporal features such as rate of change, rolling context, and anomaly scores. A supervised multiclass model predicts normal, warning, or fire states, while an anomaly detector preserves a safety-first response similar to embedded threshold logic. This makes the system suitable both for software simulation and later deployment with simple low-cost hardware.

## Dataset Strategy

This repository now supports a unified dataset created from:

- your sensor log files: `carton`, `clothing`, `electrical`
- Algerian fire-weather data
- UCI forest fire data
- optional synthetic augmentation to match your embedded sensor setup

The sensor logs are the most important source because they are closest to your embedded hardware behavior.
