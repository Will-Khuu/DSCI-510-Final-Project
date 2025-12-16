# Import necessary libraries
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords

# Configuration of file paths
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'

RAW_FILE = 'dating_apps_reviews.json'
PROCESSED_FILE = 'dating_apps_reviews_processed.csv'

FULL_RAW_PATH = os.path.join(RAW_DATA_PATH, RAW_FILE)
FULL_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILE)

# Remove the stopwords to increase the efficiency of text analysis
try:
    STOPWORDS_SET = set(stopwords.words('english'))
except LookupError:
    print("NTLK stopwords not found")
    STOPWORDS_SET = set()


# HELPER FUNCTIONS
def clean_text(text):
    """
    Cleans the input text by removing special characters, numbers, and stopwords.
    
    Parameters:
    text (str): The text to be cleaned.
    
    Returns:
    str: The cleaned text.
    """

    if not isinstance(text, str):
        return ""
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in STOPWORDS_SET])
    # Remove URLS
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove extra spaces
    text = re.sub(' +', ' ', text).strip()
    
    return text

def create_feature_flags(df):
    """
    Creates binary feature flags (1/0) based on the presence of specific keywords in the 'content' column."""
    # Flag 1: Critical Trust/Safety Issues:
    df['flag_safety'] = df['content'].str.contains('scam|fraud|unsafe|dangerous|harassment|abuse|security', case=False, na=False).astype(int)

    # Flag 2: Subscription/Payment Complaints:
    df['flag_subscription'] = df['content'].str.contains('subscription|pay|plus|HingeX', case=False, na=False).astype(int)

    return df

def main():
    df = None
    try:
        df = pd.read_json(FULL_RAW_PATH, lines=True)
        print("Loaded data successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if df is None or df.empty:
        print("No data to process.")
        return
    
    df.dropna(subset=['content'], inplace=True)
    df = create_feature_flags(df.copy())
    print(f'Dropped rows with missing content and created feature flags. Remaining rows: {len(df)}')

    df['cleaned_content'] = df['content'].apply(clean_text)
    print("Cleaned text data.")

    df['review_date'] = pd.to_datetime(df['at'])

    df['review_length'] = df['cleaned_content'].apply(lambda x: len(x.split()))

    final_columns = ['app_name', 'review_date', 'score', 'cleaned_content', 'review_length', 'flag_safety', 'flag_subscription']

    df_processed = df[final_columns].rename(columns={'score': 'rating'})

    os.makedirs(os.path.dirname(FULL_PROCESSED_DATA_PATH), exist_ok=True)
    df_processed.to_csv(FULL_PROCESSED_DATA_PATH, index=False)

    print(f"Processed data saved to {FULL_PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()