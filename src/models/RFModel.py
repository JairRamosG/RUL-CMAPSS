from sklearn.ensemble import RandomForestRegressor
from src.models.sklearn_wrapper import SKLearnModel


class RFModel(SKLearnModel):
    """Random Forest Regresssor para predicción de RUL."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
    ):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "n_jobs": n_jobs,
}
        model = RandomForestRegressor(**params)
        super().__init__(model=model)
