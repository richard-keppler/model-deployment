# NYC Taxi Trip Duration – Model Deployment

This project trains a model that predicts the duration of NYC Yellow Taxi trips (January 2025),
tracks experiments with MLflow and serves predictions through a FastAPI app.
The API runs locally, in Docker and publicly on Render.

The original assignment is described in [project-description.md](project-description.md).

## Live API

The API is deployed on Render (free tier):

- **Base URL:** https://taxi-duration-api.onrender.com
- **Interactive docs:** https://taxi-duration-api.onrender.com/docs

> **Note:** The free instance sleeps after ~15 minutes without traffic.
> The first request after that can take up to a minute while the service wakes up.

### Example request

```bash
curl -X POST https://taxi-duration-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PULocationID": 142,
    "DOLocationID": 236,
    "trip_distance": 3.2,
    "tpep_pickup_datetime": "2025-01-10T08:00:00"
  }'
```

Example response:

```json
{"predicted_duration_minutes": 13.9}
```

The exact value depends on the deployed model.

### Input fields

| Field | Type | Description |
|---|---|---|
| `PULocationID` | int (1–265) | TLC taxi zone where the trip starts |
| `DOLocationID` | int (1–265) | TLC taxi zone where the trip ends |
| `trip_distance` | float (>0, ≤100) | Trip distance in miles |
| `tpep_pickup_datetime` | datetime | Pickup time, ISO 8601 (e.g. `2025-01-10T08:00:00`) |

Invalid input returns HTTP `422` with a description of the problem.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check, returns `{"status": "ok"}` |
| `POST` | `/predict` | Returns the predicted trip duration in minutes |

## Project Structure

| Path | Description |
|---|---|
| `src/data.py` | Downloads and loads the raw parquet data |
| `src/preprocessing.py` | Cleans the data, builds features, splits train/test |
| `src/train.py` | Trains the Random Forest and a baseline, logs the run to MLflow |
| `src/model.py` | Loads the model from MLflow or a local folder (via `MODEL_URI`) |
| `app/main.py` | FastAPI app with `/` (health check) and `/predict` |
| `artifacts/model/` | Exported model used by the Docker image |
| `sample-request.json` | Example request body |
| `Dockerfile` | Container image for the API |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

## Running Locally

### 1. Start the MLflow server

```bash
uv run mlflow server --port 5001
```

> Port 5000 is used by the AirPlay Receiver on macOS (requests return `403`),
> so this project uses port 5001. The UI is available at http://localhost:5001.

Runs are stored in `mlflow.db`, model artifacts in `mlartifacts/` (both ignored by Git).

### 2. Train the model

In a second terminal:

```bash
uv run python -m src.train
```

This downloads the data (if not already present), cleans it, trains the Random Forest
and a mean baseline, and logs parameters, metrics and the model to MLflow.

### 3. Run the API

```bash
uv run uvicorn app.main:app --reload --port 8000
```

By default the API loads the model from the MLflow server.
To load it from the local folder instead (no MLflow server needed):

```bash
MODEL_URI=artifacts/model uv run uvicorn app.main:app --port 8000
```

### 4. Send a request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample-request.json
```

## Running with Docker

```bash
docker build -t taxi-api .
docker run -p 8080:8080 taxi-api
```

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @sample-request.json
```

The image loads the model from `artifacts/model/` and does not need an MLflow server.
To ship a newly trained model, copy the model folder of the desired run from `mlartifacts/`
(the folder containing the `MLmodel` file) to `artifacts/model/`.

## Model and Results

**Data:** NYC TLC Yellow Taxi trip records, January 2025 (~3.47M trips, ~3.21M after cleaning)

**Features:** `PULocationID`, `DOLocationID`, `trip_distance`, `pickup_hour`, `pickup_dayofweek`
(only information that is known before the trip starts)

**Target:** `duration` in minutes (dropoff time − pickup time)

**Data cleaning:**

- pickup and dropoff within January 2025
- trip distance greater than 0 and below 100 miles
- duration between 1 and 60 minutes
- non-negative fare
- average speed below 80 mph

**Split:** 80 % train / 20 % test, `random_state=42`

**Model:** `RandomForestRegressor` with `n_estimators=50`, `max_depth=10`, `min_samples_leaf=10`,
`max_samples=0.1`, `n_jobs=-1`, `random_state=42`.
The trees are kept shallow and each tree only sees 10 % of the training data so that training
fits into 8 GB RAM on a MacBook Air (M1) and the model stays small (6.6 MB) for deployment.

| Model | Test RMSE (minutes) |
|---|---|
| Baseline (always predicts the training mean) | 9.77 |
| Random Forest | **4.39** |

The Random Forest reduces the error by about 55 % compared to the baseline.

## Questions

### 1. What is the RMSE of your final model?

The final Random Forest reaches a **test RMSE of 4.39 minutes**
(baseline: 9.77 minutes).

### 2. What would you do differently if you had more time?

- **Hyperparameter tuning** with Optuna (`max_depth`, `min_samples_leaf`, `n_estimators`,
  `max_samples`), logging every trial to MLflow, and checking whether the gain justifies a larger model.
- **Use more of the training data**: each tree currently sees only 10 % of the data because of memory limits.
- **Better features**: treat the zone IDs as categories (or use the zone-to-borough mapping),
  add the pickup–dropoff zone pair, holidays and rush-hour flags.
- **Try other models** such as gradient boosting (e.g. `HistGradientBoostingRegressor`),
  which is usually faster and more accurate on tabular data.
- **Tests** for preprocessing, model loading and the API (FastAPI `TestClient`).
- **Slimmer Docker image**: split training-only dependencies (plotly, seaborn, optuna,
  great-expectations) from the API dependencies and load the model without MLflow.
- **Model registry instead of committing the model**: use the MLflow Model Registry or cloud storage
  so a new model can be deployed without changing the Git repository.
- **Validate on later months** (e.g. February 2025) to check how well the model generalises over time.

## Deployment Notes

The trained model (6.6 MB) is committed to the repository in `artifacts/model/`
so that Render can build the Docker image directly from GitHub.
Larger models would be stored outside of Git (e.g. a model registry or cloud storage).
