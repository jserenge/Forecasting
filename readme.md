# ⚡ Forecasts App with SARIMAX

## 🚀 Overview
The **Forecasts App with SARIMAX** is a time series forecasting application built using **Streamlit**. It enables users to analyze historical data and generate future predictions using the **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)** model.

## 1️⃣ Key Features
- ⬇ Upload your own dataset (Excel format)
- 🔄 Automatic date feature extraction
- ⚖ Select a feature for forecasting
- 🗒 Adjustable forecast period (1-30 days)
- 🌐 Interactive visualizations
- 📅 Downloadable forecast results

## 2️⃣ How It Works
### 📄 Data Processing
1. User uploads an **Excel file** with a **date column** and numerical features.
2. The app converts the date column to **datetime format** and extracts **Year, Month, Day, and Day of the Week**.
3. The user selects a **feature to forecast**.
4. Data is preprocessed, ensuring valid numerical values for modeling.

### 🔢 Forecasting with SARIMAX
1. A **SARIMAX model** is trained on the selected feature.
2. The model forecasts the next **N days** based on user input.
3. Prediction intervals (margin of error) are computed.

### 📊 Visualization
- The app plots **actual vs. forecasted values**.
- A shaded region represents the **margin of error**.

## 3️⃣ Application Structure
### 🏰 Components
- **File Upload Module**: Handles user file input.
- **Date Feature Extraction**: Prepares time-related variables.
- **SARIMAX Model Training**: Fits the forecasting model.
- **Forecast Visualization**: Displays predictions interactively.

### 📂 Key Code Snippets
#### 🔄 Date Processing
```python
def create_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    return df
```

#### 🌐 Forecasting with SARIMAX
```python
def fit_and_predict(df, selected_feature, num_days, margin_of_error=0.05):
    model = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=num_days)
    return forecast.summary_frame()
```

## 4️⃣ Installation & Usage
### 🛠️ Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### ▶ Running the App
```bash
streamlit run main.py
```

## 5️⃣ Future Enhancements
- ✨ **Hyperparameter tuning for better accuracy**
- 📊 **Additional forecast models for comparison**
- 🌐 **Cloud deployment for accessibility**

📊 Make data-driven decisions with accurate forecasts! ⚡
