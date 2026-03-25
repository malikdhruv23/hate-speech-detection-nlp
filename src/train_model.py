import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load processed data
df = pd.read_csv("data/processed_data.csv")
df = df.dropna(subset=['clean_text'])
# Features and target
X = df['clean_text']
y = df['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model
model = LogisticRegression(max_iter=200)

model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

import pickle
import os

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model and TF-IDF saved!")