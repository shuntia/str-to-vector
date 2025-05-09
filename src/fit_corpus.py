from sklearn.datasets import fetch_20newsgroups
from vector_pipeline import VectorPipeline
import joblib
from datasets import load_dataset, concatenate_datasets
import nltk


imdb_dataset = load_dataset("imdb")
corpus = imdb_dataset["train"]["text"] + imdb_dataset["test"]["text"]

def fit_umap():
# corpus = fetch_20newsgroups(remove=("headers", "footers", "quotes")).data


    pipe = VectorPipeline(n_components=3)
    pipe.fit(corpus)
    joblib.dump(pipe, "umap_prefit2.pkl")

def fit_pca():
    pipe = VectorPipeline(n_components=3)
    pipe.fit(corpus)
    joblib.dump(pipe, "pca_prefit.pkl")

def fit_umap_wikitext():
    print("Loading Wikitext-103 dataset...")
    # Load all splits of wikitext-103
    dataset_train = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    dataset_validation = load_dataset('wikitext', 'wikitext-103-v1', split='validation')
    dataset_test = load_dataset('wikitext', 'wikitext-103-v1', split='test')

    all_wikitext_data = concatenate_datasets([dataset_train, dataset_validation, dataset_test])
    
    print(f"Loaded {len(all_wikitext_data)} paragraphs/sections from Wikitext-103.")

    corpus_sentences = []
    print("Tokenizing into sentences (this may take a while)...")
    for i, item in enumerate(all_wikitext_data['text']):
        if item.strip(): # Ensure the text is not just whitespace
            sentences = nltk.sent_tokenize(item)
            corpus_sentences.extend([s for s in sentences if len(s.split()) > 3]) # Filter very short sentences
        if (i + 1) % 10000 == 0: # Progress update
            print(f"Processed {i+1} paragraphs, collected {len(corpus_sentences)} sentences...")

    print(f"Total sentences for fitting: {len(corpus_sentences)}")

    # --- Crucial: Ensure your VectorPipeline is configured to use UMAP ---
    # Example: In vector_pipeline.py, UMAP should be active and PCA commented out
    # steps = [
    #     ("embed", Distilberta()),
    #     # ("pca", PCA(n_components=n_components, random_state=42)), 
    #     ( 
    #         "umap",
    #         umap.UMAP(
    #             n_components=3, # For 3D visualization
    #             n_neighbors=15, # Default is 15, can tune
    #             min_dist=0.1,   # Default is 0.1, can tune
    #             metric="cosine",
    #             random_state=42,
    #             verbose=True # Good for long fits
    #         ),
    #     ),
    # ]
    # --------------------------------------------------------------------

    print("Initializing VectorPipeline with UMAP (n_components=3)...")
    pipe = VectorPipeline(n_components=3) 

    print("Fitting UMAP on the Wikitext sentences (this will take a significant amount of time)...")
    pipe.fit(corpus_sentences[:10000]) 
    
    output_filename = "umap_wikitext.pkl"
    print(f"Saving fitted pipeline to {output_filename}...")
    joblib.dump(pipe, output_filename)
    print("Fitting complete and pipeline saved.")

if __name__ == "__main__":
    fit_umap_wikitext()
