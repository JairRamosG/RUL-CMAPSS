from sklearn.svm import SVR
from src.models.sklearn_wrapper import SKLearnModel


class SVRModel(SKLearnModel):
    """SVR con kernel RBF para predicción de RUL."""
    
    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: str = "rbf",
    ):
        params = {"C": C, "epsilon": epsilon, "kernel": kernel}
        model = SVR(**params)
        super().__init__(model=model)
