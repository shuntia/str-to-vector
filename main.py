from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def main():
    df = pd.DataFrame({"raw_texts": [], "embeddings": []})
    text = input("Enter text to vectorize: ").splitlines()
    vectorized = vectorize(text)
    vectorized_df = pd.DataFrame(
        {"raw_texts": text, "embeddings": [vec for vec in vectorized.toarray()]}
    )
    df = pd.concat([df, vectorized_df], axis=0, ignore_index=True)
    print(df.head())


def vectorize(text):
    vectorizer = TfidfVectorizer()
    vectorized = vectorizer.fit_transform(text)
    return vectorized


if __name__ == "__main__":
    main()
