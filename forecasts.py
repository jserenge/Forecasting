import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

def create_date_features(df):
    """Create date features from the date column."""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)  # Drop rows where Date conversion failed
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    return df
def fit_and_predict(df, selected_feature, num_days, model_fit=None):
    """Fit a SARIMAX model and make forecasts."""
    df = df.rename(columns={'Date': 'ds', selected_feature: 'y'})
    
    # Ensure the 'y' column is numeric
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(subset=['y'], inplace=True)  # Drop rows where 'y' conversion failed
    
    if model_fit is None:
        # Fit the SARIMAX model
        model = SARIMAX(df['y'], 
                        order=(1, 1, 1),  # Change these orders as necessary
                        seasonal_order=(1, 1, 1, 12))  # Change seasonal order as needed
        model_fit = model.fit(disp=False)
    else:
        # Update the model with new data
        model_fit = model_fit.append(df['y'], refit=True)
    
    # Generate future dates
    future_dates = pd.date_range(start=df['ds'].max(), periods=num_days + 1, inclusive='right')
    
    # Forecast
    forecast = model_fit.get_forecast(steps=num_days)
    forecast_df = forecast.summary_frame()
    forecast_df['ds'] = future_dates
    forecast_df.rename(columns={'mean': 'yhat', 'mean_ci_lower': 'yhat_lower', 'mean_ci_upper': 'yhat_upper'}, inplace=True)
    
    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], model_fit


def app():
    st.title('Forecasts App with SARIMAX and Adaptive Learning')

    # About section
    st.sidebar.header('About')
    st.sidebar.write("""
    This application uses the SARIMAX model to forecast future values based on historical data. Here's how it works:
    
    1. **Upload Your Data**: Upload an Excel file containing your historical data. The file should have a column with date information and at least one column with the values you want to forecast.
    
    2. **Create Date Features**: The app will automatically create date-related features from the date column, such as year, month, day, and day of the week.
    
    3. **Select the Feature to Forecast**: Choose which column of your data you want to use for forecasting. This column should contain the values you wish to predict.
    
    4. **Specify Forecast Settings**: Enter the number of days you want to forecast into the future. The app will generate forecasts for this period.
    
    5. **View and Download Forecast Results**: The app will display the forecast results, including the predicted values and their confidence intervals. You can also download the results in Excel format.
    
    **Model Details**:
    - The SARIMAX model is used to capture seasonality and trend in the data.
    - The model parameters may be adjusted based on the characteristics of your data.
    - The model now includes adaptive learning to update its parameters with new data.
    """)
    
    st.write('Upload your data and specify the forecast settings below.')

    # File upload
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    
    if uploaded_file:
        try:
            # Read the uploaded file
            df = pd.read_excel(uploaded_file)
            
            # Show the uploaded data
            st.subheader("Uploaded Data")
            st.dataframe(df)

            # Feature selection
            selected_feature = st.selectbox("Select the feature to forecast", df.columns)

            # Create date features
            df = create_date_features(df)

            # Slider input
            num_days = st.slider("Number of days to forecast", min_value=1, max_value=30, value=7)
            
            if st.button('Generate Forecast'):
                with st.spinner('Generating forecast...'):
                    try:
                        forecast_df, model_fit = fit_and_predict(df, selected_feature, num_days)

                        st.subheader('Forecast Results:')
                        st.dataframe(forecast_df)

                        # Save the model for future use
                        st.session_state['model_fit'] = model_fit

                    except Exception as e:
                        st.error(f"An error occurred during forecasting: {e}")

        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

if __name__ == '__main__':
    app()
