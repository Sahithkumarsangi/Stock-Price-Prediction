# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from flask import Flask, render_template, request
import joblib
from io import BytesIO
import base64

app = Flask(__name__)

# Load pre-trained models
linear_regression_model = joblib.load('mymodels/linear_regression_model.pkl')
lstm_model = tf.keras.models.load_model('mymodels/lstm_model.h5')

# Function to compute Relative Strength Index (RSI)
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Feature Engineering: Creating technical indicators
def add_technical_indicators(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df.dropna(inplace=True)
    return df

# Function to predict using Linear Regression
def predict_linear_regression(df):
    X = df[['MA10', 'MA100', 'EMA10', 'RSI', 'Volume']]
    y_test = df['Close']
    
    y_pred = linear_regression_model.predict(X)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, mae, r2, y_test, y_pred,y_test.index

# Function to predict using pre-trained LSTM model
def predict_lstm(df):
    features = ['Close']
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize the data
    scaled_data = scaler.fit_transform(data)
    
    # Create test data
    def create_sequences(data, seq_length=120):
        X = []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i, 0])
        return np.array(X)
    
    X_test = create_sequences(scaled_data)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Predict using LSTM model
    predictions = lstm_model.predict(X_test)

    # Inverse transform the predictions
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(scaled_data[120:])

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return mse, mae, r2, y_test.flatten(), predictions.flatten()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock_name']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    model_choice = request.form.get('model')
    
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    stock_data = add_technical_indicators(stock_data)

    if model_choice == 'linear_regression':
        mse, mae, r2, y_test, y_pred,y_test_index = predict_linear_regression(stock_data)
        y_test = y_test.values
        
    elif model_choice == 'lstm':
        mse, mae, r2, y_test, y_pred = predict_lstm(stock_data)

    # Create the graphs
    # 1. Closing Price vs Time
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'], label='Closing Price', color='blue')
    plt.title(f'{stock_name} - Closing Price vs Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    img1 = BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    closing_price_plot_url = base64.b64encode(img1.getvalue()).decode()

    # 2. Closing Price with 100-day and 200-day MA
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index, stock_data['Close'], label='Closing Price')
    
    # Plot 100-day MA (only where 100-day MA exists)
    if 'MA100' in stock_data.columns:
        plt.plot(stock_data.index[99:], stock_data['MA100'][99:], label='100-day MA', color='orange')
    
    # Plot 200-day MA (only where 200-day MA exists)
    if 'MA200' in stock_data.columns:
        plt.plot(stock_data.index[199:], stock_data['MA200'][199:], label='200-day MA', color='green')
    plt.title(f'{stock_name} - Closing Price with 100-Day & 200-Day MA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    img3 = BytesIO()
    plt.savefig(img3, format='png')
    img3.seek(0)
    closing_price_100_200ma_plot_url = base64.b64encode(img3.getvalue()).decode()

    # 3. Actual vs Model Predicted
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_pred, label=f'{model_choice.capitalize()} Prediction')
    plt.title(f'{stock_name} - {model_choice.capitalize()} Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    img4 = BytesIO()
    plt.savefig(img4, format='png')
    img4.seek(0)
    predicted_vs_actual_plot_url = base64.b64encode(img4.getvalue()).decode()

    # Pass the metrics (MAE, MSE, R²) to the template
    return render_template('result.html', 
                           mse=mse, 
                           mae=mae, 
                           r2=r2,
                           closing_price_plot_url=closing_price_plot_url,
                           closing_price_100_200ma_plot_url=closing_price_100_200ma_plot_url,
                           predicted_vs_actual_plot_url=predicted_vs_actual_plot_url)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)







