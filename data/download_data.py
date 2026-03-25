import pandas as pd
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"

df = pd.read_csv(url)

print("Shape of dataset:", df.shape)

df.to_csv("data/labeled_data.csv", index=False)

print("✅ Dataset downloaded and saved successfully!")