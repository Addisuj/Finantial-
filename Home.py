import streamlit as st
import pandas as pd
import re
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="News Analysis Dashboard", layout="wide")

# Sidebar File Uploader
with st.sidebar:
    upload_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if not upload_file:
        st.warning("Please upload a file to proceed.")
        st.stop()

# Load the data
if upload_file.name.endswith(".csv"):
    data = pd.read_csv(upload_file)
elif upload_file.name.endswith(".xlsx"):
    data = pd.read_excel(upload_file)

# Data Transformation
data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Convert to datetime

# Data Cleaning
data = data.drop_duplicates()  # Remove duplicates
data = data.dropna()  # Handle missing values

# Text Normalization
data['headline'] = data['headline'].str.lower()  # Convert to lowercase
data['headline'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation

# Extract additional features
data['headline_length'] = data['headline'].apply(len)  # Headline length
data['day_of_week'] = data['date'].dt.day_name()  # Day of the week
data['month'] = data['date'].dt.month_name()  # Month
data['year'] = data['date'].dt.year  # Year

# Data Integrity Checks
def is_valid_url(url):
    return re.match(r'^(https?://)?([\da-z\.-]+)\.([a-z\.]{2,6})([/\w \.-]*)*/?$', url) is not None

data['url_valid'] = data['url'].apply(is_valid_url)  # Validate URLs

# Descriptive Statistics
st.write("Descriptive Statistics:")
st.write("Textual Length (Headline Length):")
st.write(data['headline_length'].describe())

st.write("Number of Articles per Publisher:")
publisher_counts = data['publisher'].value_counts()
st.bar_chart(publisher_counts)

# Text Analysis (Sentiment Analysis & Topic Modeling)
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_category(text):
    sentiment_score = analyzer.polarity_scores(text)['compound']
    if sentiment_score > 0.05:
        return "Positive"
    elif sentiment_score < -0.05:
        return "Negative"
    else:
        return "Neutral"

data['sentiment'] = data['headline'].apply(get_sentiment_category)  # Sentiment analysis

st.write("Sentiment Distribution:")
sentiment_counts = data['sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# Topic Modeling
cleaned_headlines = data['headline'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

vectorizer = CountVectorizer(stop_words='english', max_features=1000)
word_counts = vectorizer.fit_transform(cleaned_headlines)

lda = LatentDirichletAllocation(n_components=5, random_state=42)  # 5 topics
lda.fit(word_counts)

# Function to extract topics
def get_topics(model, feature_names, num_top_words):
    topics = []
    # Safely extract topics using 'components_' without parentheses
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        topics.append(" ".join(topic_words))
    return topics

feature_names = vectorizer.get_feature_names_out()
topics = get_topics(lda, feature_names, 10)  # Extract top 10 words per topic

st.write("Identified Topics:")
for i, topic in enumerate(topics):
    st.write(f"Topic {i + 1}: {topic}")

# Time Series Analysis
st.write("Publication Frequency Over Time:")
data['date_only'] = data['date'].dt.date  # Extract date without time
pub_freq = data['date_only'].value_counts().sort_index()

st.line_chart(pub_freq)  # Line chart showing publication frequency over time

# Publisher Analysis
# Ensure proper domain extraction to avoid IndexError
def extract_domain(publisher):
    split_result = re.split(r'@', publisher)
    if len(split_result) > 1:
        domain = split_result[1]
        domain_split = re.split(r'\.', domain)
        if len(domain_split) > 1:
            return domain_split[-2]  # Return the domain
    return "unknown"  # Default value if split fails

publisher_domains = data['publisher'].apply(extract_domain)  # Extract domains
domain_counts = publisher_domains.value_counts()

st.write("Top Publishing Organizations:")
st.bar_chart(domain_counts.head(10))  # Bar chart showing top publishing organizations
