import xgboost as xgb
from src.models.sklearn_wrapper import SKLearnModel


class XGBoostModel(SKLearnModel):
    """XGBoost para predicción de RUL."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        n_jobs: int = -1,
        ):

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "n_jobs": n_jobs
        }

        model = xgb.XGBRFRegressor(**params)
        super().__init__(model=model)
