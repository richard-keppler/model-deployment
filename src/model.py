import mlflow
import os

TRACKING_URI = os.getenv("TRACKING_URI", "http://localhost:5001")
MODEL_URI = os.getenv("MODEL_URI", "models:/m-58bd4051a1c4447eb0d7a10fc49234b1")


def load_model(tracking_uri=TRACKING_URI, model_uri=MODEL_URI):
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(model_uri)
    return model