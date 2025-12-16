# Analyzing User Sentiments Across Popular Dating Apps
This project performs a comprehensive, data-driven analysis of user reviews for the three leading dating applications: Tinder, Bumble, and Hinge. By employing VADER Sentiments and Latent Dirichlet Allocation (LDA) Topic Modelling, we quantify overall user satisfaction and identify the key thematic drivers behind user complaints and praise.

# Project Setup

### Installation

To set up the project environment, you must run the following command to install all necessary Python libraries:

`pip install -r requirements.txt`

### Running the Analysis Pipeline

The entire project pipeline, from data collection through visualizations, can be run by executing the three main scripts sequentially from the root directory of the repository

### How to Get Data
Process: The `get_data.py` script uses the google-play-scraper library to connect directly to the Google Play Store. It will then collect the 1,000 most recent user reviews for each of the three targeted apps. The raw reviews are saved as a single JSON file.

### How to Clean Data
Process: The `clean_data.py` script takes raw data and performs essential preprocessing steps:
- Removes blank content
- Converted text to lowercase and removed punctuation
- Eliminates common English words (i.e., 'the', 'a', 'is')
- Creates two binary flags: safety and subscription
The resulting dataset is saved as a single CSV file.

### How to Run Analysis Code
Process: The `run_analysis.py` script loads the processed data and executes the primary analytical models:
- VADER Sentiment Analysis
- Latent Dirichlet Allocation
The resulting dataframe from the analysis is used to produce the 4 visualizations in the same `run_analysis.py` script:
- Sentiment Distribution by app
- Average Rating Comparison
- Topic Distribution by app
- Feature Flag Analysis

## Project Member:
William Khuu
wkhuu@usc.edu
Will-Khuu
3398086799

