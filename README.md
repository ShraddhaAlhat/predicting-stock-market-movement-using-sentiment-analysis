

#   Predicts Stock Movements
## Project Overview


Predicts Stock Movements is a data-driven project that analyzes Reddit discussions about NVIDIA and combines them with historical stock price trends to predict potential stock movements. It uses web scraping, sentiment analysis, and technical stock indicators to generate actionable insights for stock predictions.


## Features

- Scrapes Reddit posts for NVIDIA-related discussions and extracts relevant information.
- Performs sentiment analysis on the extracted posts using VADER Sentiment Analysis.
- Merges sentiment analysis results with NVIDIA's historical stock data.
- Computes technical indicators such as RSI, EMA, SMA, Bollinger Bands, and Momentum.
- Predicts stock price movements based on combined data using a trained machine learning model.




## Project Structure
### Files and Their Purpose
1.```requirements.txt```

Contains all dependencies required for the project.

#### Dependencies:
- pandas
- numpy
- praw
- nltk
- vaderSentiment
- yfinance
- scikit-learn
- joblib
  
2.```Jupyter_Notebooks_Scripts.ipynb``` 

A notebook containing step-by-step implementation of the project:

- Scraping Reddit posts.
- Performing sentiment analysis.
- Computing stock indicators.
- Training the machine learning model.

3.```nvidia_stock_scraper.py``` 

A standalone Python script for scraping Reddit data related to NVIDIA. It collects post details such as:
- Title, text, upvotes, and comment counts.
- Saves data in a JSON file named ```nvidia_stock_posts```.json.

4.```nvidia_stock_sentiment_analysis.py```
- Preprocesses and cleans textual data from Reddit.
- Conducts sentiment analysis and saves results.
- Merges the sentiment data with historical stock - prices from Yahoo Finance.
- Computes stock indicators such as RSI, EMA, SMA, and Bollinger Bands. 

5.```model_training.py``` 
- Prepares the dataset for model training.
- Trains a machine learning model (e.g., Random Forest) to predict stock momentum.
- Saves the trained model and scaler using joblib.

6.```user_input_predict.py``` 

A script allowing users to:
- Input features such as sentiment scores, stock prices, and technical indicators.
- Standardize the input using the same scaler used during training.
- Predict stock momentum using the trained machine learning model.


## How to Set Up the Project
### Prerequisites

- Python 3.7 or later.
- Reddit API credentials for PRAW integration.
    
    #### Obtain Reddit API Credentials:

    Steps to Add Your Reddit API Credentials
      
    - Visit the [Reddit API apps page](https://www.reddit.com/prefs/apps).
    - Log in with your Reddit account.
    - Click on "Create App" or "Create Another App".
    - Fill out the form
    - Submit to get your Client ID and Client Secret.


- NVIDIA stock data via yfinance.

### Setup Instructions

1.Clone the repository:
```bash
git clone https://github.com/ShraddhaAlhat/predicting-stock-market-movement-using-sentiment-analysis.git
cd predicts-stock-movements
```

2.Install dependencies:
```bash
pip install -r requirements.txt

```
3.Configure Reddit API:

- Add your ```client_id```,```client_secret```, and ```user_agent``` to the scraper script.

## How to Run the Project

#### 1.Data Scraping

Fetch Reddit discussions about NVIDIA:

```bash
python nvidia_stock_scraper.py

```
#### 2.Data Processing & Sentiment Analysis

Analyze sentiment and compute technical indicators:

```bash
python nvidia_stock_sentiment_analysis.py

```
#### 3.Model Training

Train the machine learning model to predict stock momentum:

```bash
python model_training.py

```
#### 4.User Input & Prediction
```bash
python user_input_predict.py

```


## User Input and Prediction Workflow
The ```user_input_predict.py``` script allows users to input features manually for prediction.

### Features:

- Sentiment scores (```positive```,```neutral```, ```negative```,```compound```).
- Reddit metrics (```upvotes```, ```num_comments```).
- Stock market data (```Open```, ```High```,``` Low```, ```Close```,``` Adj Close```,``` Volume```).
- Technical indicators (```EMA_50```,``` SMA_5```, ```RSI_14```,``` Bollinger_MA```, ```Bollinger_STD```, ```Bollinger_Upper```,``` Bollinger_Lower```, ```LogMomentum```).

### Workflow:

1. Enter feature values when prompted.

2. The script:

- Collects the input data and standardizes it using a pre-saved scaler (```scaler.pkl```).
-  Predicts stock momentum using the trained model  (```trained_model.pkl```).

3.Output:

- Displays the predicted momentum:

     - **1**: Upward momentum.
     - **0**: Downward momentum.

## Directory Structure

```predicts-stock-movements/
│
├── requirements.txt                # Dependencies
├── Jupyter_Notebooks_Scripts.ipynb # Step-by-step code walkthrough
├── nvidia_stock_scraper.py         # Reddit data scraper
├── nvidia_stock_sentiment_analysis.py # Sentiment analysis and stock indicators
├── stock_prediction_model_training.py # Machine learning model training
├── user_input_predict.py           # Predict stock movements from user input
├── trained_model.pkl               # Saved Random Forest model
├── scaler.pkl                      # Saved MinMaxScaler for input data
└── data/
    ├── nvidia_stock_posts.json     # Scraped Reddit posts
    └── stock_analysis.csv          # Data ready for training```
