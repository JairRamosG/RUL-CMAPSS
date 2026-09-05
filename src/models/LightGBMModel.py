import lightgbm as lgb
from src.models.sklearn_wrapper import SKLearnModel


class LightGBMModel(SKLearnModel):
    """Random Forest Regresssor para predicción de RUL."""
    
    def __init__(
            self,
            n_estimators: int = 100,
            num_leaves: int = 31,
            learning_rate: float = 0.1,
            subsample: float = 0.8,
            n_jobs: int = -1,
            ):
        
        params = {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "n_jobs": n_jobs,
            }
        model = lgb.LGBMRegressor(**params)
        super().__init__(model=model)
