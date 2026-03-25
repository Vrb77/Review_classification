import re
import streamlit as st
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from keras.models import load_model


# ── IMPORTANT: These classes must be defined BEFORE joblib.load() ──
# They must also match EXACTLY what was used when saving the .joblib file.

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [self._clean(text) for text in X]
    def _clean(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

class TextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(X)
        return self
    def transform(self, X):
        return self.tfidf.transform(X).toarray()


# ── Page config ──
st.set_page_config(page_title="Review Classification")
st.title("Positive and Negative Review Classification Project")
st.subheader("By Vaishnavi Badade")

# ── Load pipeline and model ──
pre = joblib.load("text_classification_pre.joblib")
model = load_model("TextClassification.keras")

# ── UI ──
review = st.text_input("Enter your review")
submit = st.button("Predict Sentiment")

if submit:
    if not review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        transformed_text = pre.transform([review])
        preds = model.predict(transformed_text)
        if preds[0][0] > 0.5:
            st.subheader("✅ Positive Review")
        else:
            st.subheader("❌ Negative Review")
