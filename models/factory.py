from models.cnn import CNN
from models.mlp import MLP


def create_model(
        model_name,
        num_classes=10, #numero de clases de salida
):
    if model_name == "mlp":
        return MLP()
    if model_name == "cnn":
        return CNN()
    raise ValueError(f"Unknown model name: {model_name}")