import pandas as pd
import re
import nltk

# Download once
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv("data/labeled_data.csv")

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Cleaning function
def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    
    # Remove special characters
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Remove stopwords + lemmatization
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply cleaning
df['clean_text'] = df['tweet'].apply(clean_text)

# Save processed data
df.to_csv("data/processed_data.csv", index=False)

print("✅ Text preprocessing completed!")
print(df[['tweet', 'clean_text']].head())