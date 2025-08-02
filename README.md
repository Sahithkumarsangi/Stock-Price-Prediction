
# 📈 Stock Price Prediction Web Application

This project focuses on predicting stock prices using Machine Learning models like **Linear Regression** and **LSTM (Long Short-Term Memory)** neural networks. The project is presented as an interactive **web application**, where users can select a stock, define a date range, and visualize the predicted stock prices compared to actual prices.

---

## 🚀 Features
- Predict future stock prices using **Linear Regression** and **LSTM** models.
- Interactive web interface using **Flask** for user inputs.
- Visualizations for **Actual vs Predicted Prices**.
- Additional plots like **Closing Price with Moving Averages (100-day, 200-day)**.
- Performance metrics: **Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score**.
- Downloadable prediction reports (optional).

---

## 🧑‍💻 Tech Stack
- **Python** (Jupyter Notebook, Flask)
- **Machine Learning Models**: Linear Regression, LSTM (TensorFlow/Keras)
- **Libraries**: NumPy, Pandas, Matplotlib, Scikit-learn, yfinance
- **Web Deployment**: GitHub Pages / Flask Local Server

---

## 🏗️ Project Structure
```
├── app.py                         # Flask backend application
├── templates/
│   ├── index.html                  # Main user input page
│   └── result.html                 # Result visualization page
├── mymodels/
│   ├── linear_regression_model.joblib
│   └── lstm_model.h5
├── stock_data/                     # Folder for storing downloaded stock CSVs
├── notebooks/
│   └── stock_prediction_model.ipynb # Jupyter notebook for model training
├── README.md
└── requirements.txt                # Python package dependencies
```

---

## 📊 Dataset
- Stock data is retrieved live using **Yahoo Finance API (yfinance)**.
- Users provide:
  - **Stock Ticker** (e.g., TATAMOTORS.NS)
  - **Start Date** and **End Date**

Features Used:
- Closing Prices
- Moving Averages (MA10, MA100, MA200)
- RSI (Relative Strength Index)

---

## 🧠 Machine Learning Models
### Linear Regression Model:
- Predicts future closing prices based on historical trends.

### LSTM Model (Deep Learning):
- Uses Sequential LSTM layers to capture temporal dependencies in stock price data.
- Architecture:
  - LSTM layers with 100 neurons each
  - Dense layers with 50 neurons
  - Output layer with 1 neuron

### Performance Metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

---

## 📈 Visualizations
- **Closing Price vs Time Chart**
- **Closing Price with 100-Day & 200-Day Moving Averages**
- **Actual vs Predicted Prices Graphs**
- **Model Accuracy Metrics Display**

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Sahithkumarsangi/Stock-Price-Prediction.git
   cd Stock-Price-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

---

## 📝 Future Scope
- Add more advanced models like **Random Forest Regressor**.
- Deploy on cloud platforms (Heroku, AWS, Render).
- Add stock price forecasting for multiple days ahead.
- Include more technical indicators (EMA, Bollinger Bands, MACD).

---

## 📬 Contact
For questions, reach out to:  
📧 [sahithkumarsangi1807@gmail.com](mailto:sahithkumarsangi1807@gmail.com)

---
