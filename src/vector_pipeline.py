from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from embed import HFEmbed, STEmbed, Distilberta
import umap
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
class VectorPipeline(Pipeline):
    def __init__(
        self,
        n_components=3,
        n_neighbors=30,
        min_dist=0.5,
        metric="cosine",
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        steps = [
            (
                "embed",
                Distilberta(),
            ),
            # ("pca", PCA(n_components=n_components, random_state=42)),
            (
                "umap",
                umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    random_state=42,
                ),
            ),
        ]
        super().__init__(steps)

    def vectorize(self, texts):
        return self.transform(texts)


if __name__ == "__main__":
    # pipe = VectorPipeline(n_components=3)
    # texts = [
    #     "Hello world",
    #     "This is a test",
    #     "Vectorization is fun",
    # ]
    # vectors = pipe.vectorize(texts)
    # print(vectors)

    texts = [
        "The cat sat on the sunny windowsill.",
        "Quantum computers factor large integers in milliseconds.",
    ]

    vectorized = Distilberta().transform(texts)
    sim = np.dot(vectorized[0], vectorized[1]) / (
        np.linalg.norm(vectorized[0]) * np.linalg.norm(vectorized[1])
    )
    print("unreduced sim", sim)

    pipe = joblib.load("pca_prefit.pkl")
    reduced = pipe.transform(texts)
    sim = np.dot(reduced[0], reduced[1]) / (
        np.linalg.norm(reduced[0]) * np.linalg.norm(reduced[1])
    )
    print("reduced sim", sim)
