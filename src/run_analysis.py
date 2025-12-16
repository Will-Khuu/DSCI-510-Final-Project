import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Configuation path
PROCESSED_DATA_PATH = 'data/processed'
PROCESSED_FILE = 'dating_apps_reviews_processed.csv'
FULL_PROCESSED_PATH = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILE)

# Initialize the VADER sentimaent analyzer
try: 
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Error initializing SentimentIntensityAnalyzer: {e}")
    sid = None

# Helper functions
def get_sentiment_vader(text):
    """
    Get VADER sentiment scores for the given text and returns a category.

    Text can be: 'positive', 'negative', or 'neutral'
    """
    if sid is None or not isinstance(text, str):
        return 'neutral'
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'
    
def run_sentiment_analysis(df):
    """
    Apply sentiment analysis to the DataFrame and creates a new column
    """

    print("Starting sentiment analysis using VADER...")
    df['sentiment'] = df['cleaned_content'].apply(get_sentiment_vader)
    print("Sentiment analysis completed.")
    return df

def plot_sentiment_distribution(df):
    """
    Plot the distribution of sentiment categories across apps
    """
    sentiment_count = df.groupby('app_name')['sentiment'].value_counts(normalize=True).mul(100).rename('proportion').reset_index()
    sns.barplot(data=sentiment_count, x='app_name', y='proportion', hue='sentiment', 
                palette={'positive': 'green', 'neutral': 'gray', 'negative': 'red'})
    plt.title('Sentiment Distribution by App')
    plt.ylabel('Proportion (%)')
    plt.xlabel('App Name')
    plt.legend(title='Sentiment')
    plt.savefig('results/sentiment_distribution_by_app.png')
    plt.close()

def display_topics(model, feature_names, no_top_words):
    """
    Display the top words for each topic in the LDA model
    """
    print("Top words per topic:")
    topic_output =['\n-- Discovered Topics (LDA --)']
    topic_summary = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        
        line = f"Topic {topic_idx}: {top_words}"
        print(line)
        topic_output.append(line)
        
        topic_summary[topic_idx] = top_words
    return topic_summary, "\n".join(topic_output)

def run_topic_modelling(df, num_topics=5, no_top_words=10):
    """
    Performs LDA on the cleaned review content
    """
    df_lda = df.copy()
    
    df_lda = df[df['cleaned_content'].str.strip() != ''].copy()
    df_lda.dropna(subset=['cleaned_content'], inplace=True)

    if len(df_lda) == 0:
        print('Warning: No content left to run LDA')
        df['topic'] = -1
        return df
    
    print(f'Starting topic modelling with {num_topics} topics')

    # Use CountVectorizer for LDA inputs
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

    # Fit and transform the cleaned content
    data_vectorized = vectorizer.fit_transform(df_lda['cleaned_content'])

    # Train LDA Model
    lda_model = LatentDirichletAllocation(n_components=num_topics,
                                          max_iter=5,
                                          learning_method='online',
                                          random_state=42,
                                          n_jobs=-1)

    lda_model.fit(data_vectorized)

    feature_names = vectorizer.get_feature_names_out()
    topic_summaries = display_topics(lda_model, feature_names, no_top_words)

    doc_topic_distribution = lda_model.transform(data_vectorized)
    df_lda['topic'] = doc_topic_distribution.argmax(axis=1)

    df = df.merge(df_lda[['topic']], left_index=True, right_index=True, how='left')
    df['topic'].fillna(-1, inplace=True)

    return df

def plot_rating_comparison(df):
    """
    Creates a bar chart comparing the average rating across the dating apps.
    """

    avg_rating = df.groupby('app_name')['rating'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=avg_rating, x = 'app_name', y='rating', palette='viridis')
    plt.title('Average User Rating by App')
    plt.xlabel('Dating App')
    plt.ylabel('Average Rating')
    plt.ylim(1, 5)
    plt.savefig('results/average_rating_comparison.png')
    plt.close()

def plot_topic_distribution(df, num_topics=5):
    """
    Creates a bar chart showing the distribution of topics across the apps

    """
    topic_map = {i: f'Topic {i}' for i in range(num_topics)}
    df['topic_label'] = df['topic'].map(topic_map)
    topic_counts = df.groupby('app_name')['topic_label'].value_counts(normalize=True).mul(100).rename('proportion').reset_index()

    sns.barplot(data=topic_counts, x='app_name', y='proportion', hue='topic_label', 
                palette='tab10')
    plt.title('Topic Distribution by App')
    plt.ylabel('Proportion (%)')
    plt.xlabel('App Name')
    plt.legend(title='LDA Topic')
    plt.savefig('results/topic_distribution_by_app.png')
    plt.close()

def plot_feature_flag_analysis(df):
    """
    Creates a bar chart comparing the proportion of safety and subscription complaints across the dating apps.
    """
    flag_comparison = df.groupby('app_name')[['flag_safety', 'flag_subscription']].mean().mul(100)
    flag_comparison = flag_comparison.stack().rename('proportion').reset_index()
    flag_comparison.columns = ['app_name', 'flag_type', 'proportion']
    flag_comparison['flag_type'] = flag_comparison['flag_type'].replace({'flag_safety': 'Safety/Scam Complaints',
                                                                         'flag_subscription': 'Subscription/Paywall Complaints'})
    plt.figure(figsize=(10, 6))
    sns.barplot(data=flag_comparison, x='app_name', y='proportion', hue='flag_type', palette =['darkorange', 'darkred'])
    plt.title('Proportion of Key Complaints by Dating App')
    plt.xlabel('Dating App')
    plt.ylabel('Percentage of Reviews (%)')
    plt.grid(axis='y', linestyle = '--')
    plt.savefig('results/feature_flag_analysis.png')
    plt.close()

# Main Function
def main():
    try:
        df = pd.read_csv(FULL_PROCESSED_PATH)
        print("Loaded processed data successfully.")
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return
    
    df = run_sentiment_analysis(df)

    df = run_topic_modelling(df, num_topics=5, no_top_words=10)

    os.makedirs('results', exist_ok=True)
    plot_sentiment_distribution(df)
    plot_feature_flag_analysis(df)
    plot_topic_distribution(df, num_topics=5)
    plot_rating_comparison(df)

if __name__ == "__main__":
    main()
