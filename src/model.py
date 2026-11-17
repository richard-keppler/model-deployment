from src.train import TRACKING_URI
MODEL_URI = "models:/m-58bd4051a1c4447eb0d7a10fc49234b1"
import mlflow


def load_model(tracking_uri=TRACKING_URI, model_uri=MODEL_URI):
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(model_uri)
    return model