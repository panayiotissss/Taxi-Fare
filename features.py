from sklearn.base import BaseEstimator, TransformerMixin


class AddFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        X['speed_kmph'] = X['Trip_Distance_km'] / X['Trip_Duration_Minutes'] * 60
        return X
