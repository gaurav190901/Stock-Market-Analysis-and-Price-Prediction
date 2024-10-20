# Stock Price Predictor using Linear Regression


## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Evaluation](#model-evaluation)
- [Future Work](#future-work)


## Overview
The **Stock Price Predictor** project implements a linear regression model to predict stock closing prices based on historical market data. This application assists investors and traders in making informed decisions by forecasting potential price movements.

## Features
- Predict closing prices using features like opening price, high, low, and volume.
- User-friendly interface for inputting stock data and obtaining predictions.
- Visualizations to understand data trends and model performance.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas` for data manipulation
  - `numpy` for numerical computations
  - `scikit-learn` for machine learning algorithms
  - `matplotlib` for data visualization
  - `joblib` for saving the trained model

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/gaurav190901/Stock-Market-Analysis-and-Price-Prediction.git
   cd Stock-Market-Analysis-and-Price-Prediction
## Install the required dependencies
pip install -r requirements.txt

## Usage
1.Train the Model: Run the script to train the linear regression model using historical stock data.
python src/train_model.py
2.Predict Stock Prices: After training, use the prediction script to input stock parameters and get the predicted closing price.
python src/predict_price.py

## Dataset
The project utilizes historical stock market data that includes the following columns:
- **Open**: The opening price of the stock.
- **High**: The highest price of the stock during the trading day.
- **Low**: The lowest price of the stock during the trading day.
- **Volume**: The number of shares traded during the day.
- **Close**: The closing price of the stock (target variable).


## Model Evaluation
The model's performance is evaluated using the following metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual values.
- **R-squared (R²)**: Indicates the proportion of variance in the target variable that can be explained by the model. Values closer to 1 indicate better predictive accuracy.

Additionally, visualizations comparing predicted prices to actual prices are generated using Matplotlib to help assess the model's effectiveness visually.

## Future Work
Future enhancements for this project may include:
- Implementing advanced machine learning models, such as Decision Trees or Random Forests, to improve prediction accuracy.
- Performing feature engineering techniques to derive additional informative features from the dataset.
- Integrating a web interface using frameworks like Flask or Streamlit for user-friendly interaction and real-time predictions.
- Adding data visualization dashboards to track stock price trends and model performance over time.
