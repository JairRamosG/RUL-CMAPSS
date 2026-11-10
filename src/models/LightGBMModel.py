import lightgbm as lgb
from src.models.sklearn_wrapper import SKLearnModel


class LightGBMModel(SKLearnModel):
    """Modelo LightGBM para predicción de RUL."""
    
    def __init__(
            self,
            n_estimators: int = 100,
            num_leaves: int = 31,
            learning_rate: float = 0.1,
            subsample: float = 0.8,
            n_jobs: int = -1,
            random_state: int = 42,
            ):
        
        params = {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "n_jobs": n_jobs,
            "random_state": random_state,
            }
        model = lgb.LGBMRegressor(**params)
        super().__init__(model=model)
