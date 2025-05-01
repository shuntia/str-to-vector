from sklearn.datasets import fetch_20newsgroups
from vector_pipeline import VectorPipeline
import joblib
from datasets import load_dataset

# corpus = fetch_20newsgroups(remove=("headers", "footers", "quotes")).data
imdb_dataset = load_dataset("imdb")
corpus = imdb_dataset["train"]["text"] + imdb_dataset["test"]["text"]

pipe = VectorPipeline(n_components=3)
pipe.fit(corpus)
joblib.dump(pipe, "umap_prefit2.pkl")
