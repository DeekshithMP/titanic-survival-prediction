# backend/predictor.py
import pickle
import pandas as pd
import numpy as np
from backend.preprocessing import prepare_features

class SurvivalPredictor:
    """
    Single source of truth predictor
    ALWAYS returns (probs, preds, risks)
    """

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.feature_names = self.model.feature_names_in_

    def predict(self, df: pd.DataFrame):
        X = prepare_features(df, self.feature_names)

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        risks = np.round((1 - probs) * 100).astype(int)

        return probs, preds, risks
