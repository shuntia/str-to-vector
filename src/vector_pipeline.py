from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from embed import SBert, Distilberta
import umap
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer


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
    pipe = VectorPipeline(n_components=3)
    texts = [
        "Hello world",
        "This is a test",
        "Vectorization is fun",
    ]
    vectors = pipe.vectorize(texts)
    print(vectors)
