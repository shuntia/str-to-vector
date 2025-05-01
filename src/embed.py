from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel


class Distilberta(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="sentence-transformers/all-distilroberta-v1"):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X, convert_to_tensor=True).cpu().numpy()


class STEmbed(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="sentence-transformers/all-distilroberta-v1"):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X, convert_to_tensor=True).cpu().numpy()


class HFEmbed(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        pooling="mean",
        device="cpu",
    ):
        self.model_name = model_name
        self.pooling = pooling
        self.device = device
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(model_name, device=self.device)
            self._hf_tok = None
            self._hf_model = None
        except Exception:
            self._st = None
            self._hf_tok = AutoTokenizer.from_pretrained(model_name)
            self._hf_model = (
                AutoModel.from_pretrained(model_name).to(self.device).eval()
            )

    def fit(self, X, y=None):
        return self

    @torch.inference_mode()
    def transform(self, X):
        if self._st:
            return self._st.encode(X, convert_to_numpy=True)

        enc = self._hf_tok(X, padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )
        out = self._hf_model(**enc).last_hidden_state

        if self.pooling == "cls":
            vec = out[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1)
            vec = (out * mask).sum(1) / mask.sum(1)

        return vec.cpu().numpy()


if __name__ == "__main__":
    texts = [
        "Hello world",
        "This is a test",
        "Vectorization is fun",
    ]
    model = HFEmbed("cardiffnlp/twitter-roberta-base-sentiment")

    print(model.transform(texts))
