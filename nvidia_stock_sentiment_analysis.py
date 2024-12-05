# Install required libraries
#!pip install pandas
#!pip install vaderSentiment
#!pip install yfinance

import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import numpy as np
from datetime import datetime

# Load JSON file into a Python list
with open("nvidia_stock_posts.json", "r") as file:
    json_data = json.load(file)

# Convert the list to a DataFrame
df = pd.DataFrame(json_data)
df = df.drop(columns=["post_id"])
df['full_text'] = df['title'] + ' ' + df['text']
df = df.drop(['title', 'text'], axis=1)

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s,]', '', text, flags=re.UNICODE)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Removes any non-alphanumeric characters
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Apply preprocessing on 'full_text' column
df['cleaned_text'] = df['full_text'].apply(preprocess_text)
df = df.drop(['full_text'], axis=1)

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment

df['sentiment_scores'] = df['cleaned_text'].apply(analyze_sentiment)

# Extract specific sentiment scores into separate columns
df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
df['positive'] = df['sentiment_scores'].apply(lambda x: x['pos'])
df['neutral'] = df['sentiment_scores'].apply(lambda x: x['neu'])
df['negative'] = df['sentiment_scores'].apply(lambda x: x['neg'])
df = df.drop(['cleaned_text', 'sentiment_scores'], axis=1)

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Group data by date and aggregate sentiment and upvotes/comments
aggregated_data = df.groupby([df['date'].dt.date]).agg({
    'upvotes': 'sum',
    'num_comments': 'sum',
    'compound': 'mean',
    'positive': 'mean',
    'neutral': 'mean',
    'negative': 'mean'
}).reset_index()

# Download historical data for NVIDIA stock
ticker = 'NVDA'
nvidia_data = yf.download(ticker, start='2010-10-23', end='2024-12-01')
nvidia_data = nvidia_data.reset_index()
nvidia_data.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Merge stock data with sentiment data
aggregated_data['date'] = pd.to_datetime(aggregated_data['date'])
nvidia_data['date'] = pd.to_datetime(nvidia_data['date'])
merged_data = pd.merge(nvidia_data, aggregated_data, on='date', how='outer')

# Clean merged data
merged_data = merged_data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

# Fill missing values in upvotes and comments with the median
for column in ['upvotes', 'num_comments']:
    merged_data[column].fillna(merged_data[column].median(), inplace=True)

# Fill missing sentiment columns with the mean
for column in ['compound', 'positive', 'neutral', 'negative']:
    merged_data[column].fillna(merged_data[column].mean(), inplace=True)

# Calculate 5-period Simple Moving Average (SMA)
merged_data['SMA_5'] = merged_data['Close'].rolling(window=5).mean()

# Calculate 50-period Exponential Moving Average (EMA)
merged_data['EMA_50'] = merged_data['Close'].ewm(span=50, adjust=False).mean()
merged_data['SMA_5'] = merged_data['SMA_5'].fillna(method='bfill')

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

merged_data['RSI_14'] = calculate_rsi(merged_data)
merged_data['RSI_14'] = merged_data['RSI_14'].fillna(method='bfill')

# Calculate Bollinger Bands
merged_data['Bollinger_MA'] = merged_data['Close'].rolling(window=5).mean()
merged_data['Bollinger_STD'] = merged_data['Close'].rolling(window=5).std()
merged_data['Bollinger_Upper'] = merged_data['Bollinger_MA'] + (merged_data['Bollinger_STD'] * 2)
merged_data['Bollinger_Lower'] = merged_data['Bollinger_MA'] - (merged_data['Bollinger_STD'] * 2)

# Fill missing values in Bollinger Bands columns
merged_data['Bollinger_MA'] = merged_data['Bollinger_MA'].fillna(method='bfill')
merged_data['Bollinger_STD'] = merged_data['Bollinger_STD'].fillna(method='bfill')
merged_data['Bollinger_Upper'] = merged_data['Bollinger_Upper'].fillna(method='bfill')
merged_data['Bollinger_Lower'] = merged_data['Bollinger_Lower'].fillna(method='bfill')

# Calculate Log Momentum
merged_data['LogMomentum'] = np.log(merged_data['Close'] / merged_data['Close'].shift(5))
merged_data['LogMomentum'] = merged_data['LogMomentum'].fillna(method='bfill')

# Save the final data to a CSV file
merged_data.to_csv('stock_analysis.csv', index=False)

print("Data has been successfully saved to 'stock_analysis.csv'")
