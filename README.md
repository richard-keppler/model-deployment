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
| `src/tune.py` | Hyperparameter tuning with Optuna, every trial logged as a nested MLflow run |
| `src/grid_search.py` | Hyperparameter tuning with `GridSearchCV` (12 combinations, 3-fold cross-validation), logged to MLflow |
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

### 3. (Optional) Tune hyperparameters with Optuna

```bash
uv run python -m src.tune
```

Runs 20 Optuna trials over `n_estimators`, `max_depth` and `min_samples_leaf`.
All trials are logged as nested runs under an `optuna-study` run in MLflow,
and the best configuration is retrained and logged as `best-model`.

As an alternative, an exhaustive grid search with `GridSearchCV`:

```bash
uv run python -m src.grid_search
```

It tries 12 combinations with 3-fold cross-validation on the training data and logs the best
configuration, its cross-validated RMSE (`CV_RMSE`), the test RMSE and the model as a `best-model` run
nested under a `grid-search` run.

### 4. Run the API

```bash
uv run uvicorn app.main:app --reload --port 8000
```

By default the API loads the model from the MLflow server.
To load it from the local folder instead (no MLflow server needed):

```bash
MODEL_URI=artifacts/model uv run uvicorn app.main:app --port 8000
```

### 5. Send a request

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

| Model | Test RMSE (minutes) | Model size |
|---|---|---|
| Baseline (always predicts the training mean) | 9.77 | – |
| Random Forest (deployed) | **4.39** | 6.6 MB |
| Random Forest tuned with Optuna (not deployed) | 4.06 | 113 MB |
| Random Forest tuned with grid search (not deployed) | 4.01 | 199 MB |

The Random Forest reduces the error by about 55 % compared to the baseline.

### Hyperparameter tuning with Optuna

20 Optuna trials searched `n_estimators` (20–100), `max_depth` (5–20) and `min_samples_leaf` (5–50),
with `max_samples=0.1` fixed. The best trial used `n_estimators=50`, `max_depth=19`, `min_samples_leaf=9`
and reached a test RMSE of **4.06 minutes**, about 7.6 % better than the deployed model.

**Was the extra complexity worth it?** Not for this deployment. The deeper trees make the model
17 times larger (113 MB), which exceeds GitHub's 100 MB file limit and is risky for the ~512 MB RAM
of the free Render instance, while the prediction only improves by about 20 seconds on average.
The smaller model is therefore kept in production; the tuned model stays in MLflow.

### Hyperparameter tuning with grid search

As a comparison, `GridSearchCV` tried every combination of `n_estimators` (20, 50), `max_depth` (10, 20)
and `min_samples_leaf` (5, 10, 25), again with `max_samples=0.1` fixed. Unlike the Optuna study, the
configuration is selected with 3-fold cross-validation on the training data, so the test set is only used
for the final evaluation. The run took about 3 minutes and used at most about 1 GB of RAM.

The best combination was `n_estimators=50`, `max_depth=20`, `min_samples_leaf=5` with a cross-validated
RMSE of 4.06 and a **test RMSE of 4.01 minutes**, slightly better than Optuna. The similar CV and test
errors show that the model does not overfit the test set. However, both `max_depth` and `min_samples_leaf`
ended up at the edge of the grid, so the optimum probably lies beyond it, at even deeper trees.
The model is already 199 MB, so the same conclusion holds: it is not deployed.

## Questions

### 1. What is the RMSE of your final model?

The deployed Random Forest reaches a **test RMSE of 4.39 minutes** (baseline: 9.77 minutes).
The best Optuna configuration reaches **4.06 minutes** and the best grid search configuration **4.01 minutes**,
but neither was deployed because of their size (see above).

### 2. What would you do differently if you had more time?

- **Size-aware tuning**: optimise RMSE and model size together (multi-objective Optuna study),
  include `max_samples` in the search and run more trials.
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
