from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer


class SBert(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X, convert_to_tensor=True).cpu().numpy()


class Distilberta(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="sentence-transformers/all-distilroberta-v1"):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X, convert_to_tensor=True).cpu().numpy()
