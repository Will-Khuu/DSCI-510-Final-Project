# Import Libraries
import pandas as pd
import json
import os

from google_play_scraper import Sort, reviews 

# Define File Paths
RAW_DATA_PATH = 'data/raw'

RAW_FILE = os.path.join(RAW_DATA_PATH, 'dating_apps_reviews.json')


# List of target apps
TARGET_APPS = [{'id': 'com.tinder', 'name': 'Tinder'},
               {'id': 'com.bumble.app', 'name': 'Bumble'},
               {'id': 'com.hinge.app', 'name': 'Hinge'}
]

# Number of reviews to fetch per app
NUM_REVIEWS = 1000

# Function to fetch reviews for a given app
def scrape_reviews(app_id, app_name, num_reviews):
    """
    Fetch reviews for a given app using google_play_scraper.
    Returns a DataFrame with the reviews and app name.
    """
    try:
        result, continuation_token = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=num_reviews)
        
        df = pd.DataFrame(result)
        df['app_name'] = app_name
        return df
    except Exception as e:
        print(f"Error fetching reviews for {app_name}: {e}")
        return pd.DataFrame()
    

# Main scraping process
def main():
    """
    Main function to scrape reviews for target apps and save to JSON.
    """
    all_reviews_dfs = []

    print("Starting review scraping...")
    for app in TARGET_APPS:
        print(f"Scraping reviews for {app['name']}...")
        
        df_app_reviews = scrape_reviews(app['id'], app['name'], NUM_REVIEWS)

        if not df_app_reviews.empty:
            all_reviews_dfs.append(df_app_reviews)
            print(f"Successfully fetched {len(df_app_reviews)} reviews for {app['name']}.")

    if all_reviews_dfs:
        final_reviews_df = pd.concat(all_reviews_dfs, ignore_index=True)
        
        # Save to JSON
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        final_reviews_df.to_json(RAW_FILE, orient='records', lines=True)
        print(f"Saved all reviews to {RAW_FILE}.")
    else:
        print("No reviews were fetched.")

if __name__ == "__main__":
    main()
