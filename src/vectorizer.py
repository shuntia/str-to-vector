class Vectorizer:
    def __init__(self):
        pass

    def vectorize(self, text):
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer()
        vectorized = vectorizer.fit_transform(text)
        return vectorized
