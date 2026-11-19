import mlflow
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.data import DATA_URL, FILE_PATH, download_data, load_data
from src.preprocessing import clean_data, split_data

NAME_EXPERIMENT = "taxi-duration"
TRACKING_URI = "http://localhost:5001"
FIXED_PARAMS = {"max_samples": 0.1, "n_jobs": -1, "random_state": 42}
PARAM_GRID = {
    "n_estimators": [20, 50],
    "max_depth": [10, 20],
    "min_samples_leaf": [5, 10, 25],
}


def train_mlflow(param_grid, x_train, y_train, x_test, y_test):
    with mlflow.start_run(run_name="best-model", nested=True):
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(**FIXED_PARAMS),
            param_grid=param_grid,
            cv=3,
            scoring="neg_root_mean_squared_error",
        )
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        cv_rmse = -grid_search.best_score_
        print(f"Best params: {grid_search.best_params_} | CV RMSE: {cv_rmse:.2f}")
        mlflow.log_params({**FIXED_PARAMS, **grid_search.best_params_})
        mlflow.log_metric("CV_RMSE", cv_rmse)
        y_pred = best_model.predict(x_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mlflow.log_metric("RMSE", rmse)
        dummy_regressor = DummyRegressor()
        dummy_regressor.fit(x_train, y_train)
        y_pred_dummy = dummy_regressor.predict(x_test)
        rmse_dummy = root_mean_squared_error(y_test, y_pred_dummy)
        mlflow.log_metric("RMSE_baseline", rmse_dummy)
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="taxi_duration",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            input_example=x_train.head(5),
        )
        print(f"Best model RMSE: {rmse:.2f} | Baseline: {rmse_dummy:.2f}")
        print(f"Model URI: {model_info.model_uri}")


def main(tracking_uri, name_experiment, data_url, file_path):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(name_experiment)
    file_path = download_data(data_url, file_path)
    df = load_data(file_path)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    with mlflow.start_run(run_name="grid-search"):
        train_mlflow(PARAM_GRID, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main(TRACKING_URI, NAME_EXPERIMENT, DATA_URL, FILE_PATH)
