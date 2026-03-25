# ─────────────────────────────────────────────────────────────────
# Run this cell to re-save the pipeline so Streamlit can load it.
# preprocessor.py must be in the same folder as this notebook.
# ─────────────────────────────────────────────────────────────────

import joblib
from sklearn.pipeline import Pipeline
from preprocessor import TextCleaner, TextTransformer

preprocessing_pipeline = Pipeline([
    ('cleaner',     TextCleaner()),
    ('transformer', TextTransformer())
])

preprocessing_pipeline.fit(x)   # x is your Review_Preprocessed series from earlier

joblib.dump(preprocessing_pipeline, "text_classification_pre.joblib")
print("Saved!")
