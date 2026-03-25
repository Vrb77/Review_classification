import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class TextCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._clean(text) for text in X]

    def _clean(self, text):
        text = text.lower()
        pattern = r"[^a-z ]"
        text = re.sub(pattern, "", text)
        return text


class TextTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(X)
        return self

    def transform(self, X):
        return self.tfidf.transform(X).toarray()
