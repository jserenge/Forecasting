import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import matplotlib.pyplot as plt

def create_date_features(df):
    """Create date features from the date column."""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)  # Drop rows where Date conversion failed
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    return df

def fit_and_predict(df, selected_feature, num_days, margin_of_error=0.05):
    """Fit a SARIMAX model and make forecasts."""
    df = df.rename(columns={'Date': 'ds', selected_feature: 'y'})
    
    # Ensure the 'y' column is numeric
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(subset=['y'], inplace=True)  # Drop rows where 'y' conversion failed
    
    # Fit the SARIMAX model
    model = SARIMAX(df['y'], 
                    order=(1, 1, 1),  # Change these orders as necessary
                    seasonal_order=(1, 1, 1, 12))  # Change seasonal order as needed
    model_fit = model.fit(disp=False)
    
    # Generate future dates
    future_dates = pd.date_range(start=df['ds'].max(), periods=num_days + 1, inclusive='right')
    
    # Forecast
    forecast = model_fit.get_forecast(steps=num_days)
    forecast_df = forecast.summary_frame()
    forecast_df['ds'] = future_dates
    
    # Calculate margin of error bounds
    forecast_df['yhat_lower'] = forecast_df['mean'] * (1 - margin_of_error)
    forecast_df['yhat_upper'] = forecast_df['mean'] * (1 + margin_of_error)
    forecast_df.rename(columns={'mean': 'yhat'}, inplace=True)
    
    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def app():
    st.title('Forecasts App with SARIMAX')

    # About section
    st.sidebar.header('About')
    st.sidebar.write("""
    This application uses the SARIMAX model to forecast future values based on historical data. Here's how it works:
    
    1. **Upload Your Data**: Upload an Excel file containing your historical data. The file should have a column with date information and at least one column with the values you want to forecast.
    
    2. **Create Date Features**: The app will automatically create date-related features from the date column, such as year, month, day, and day of the week.
    
    3. **Select the Feature to Forecast**: Choose which column of your data you want to use for forecasting. This column should contain the values you wish to predict.
    
    4. **Specify Forecast Settings**: Enter the number of days you want to forecast into the future. The app will generate forecasts for this period.
    
    5. **View and Download Forecast Results**: The app will display the forecast results, including the predicted values and their bounds based on a margin of error. You can also download the results in Excel format.
    
    **Model Details**:
    - The SARIMAX model is used to capture seasonality and trend in the data.
    - The model parameters may be adjusted based on the characteristics of your data.
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
                        forecast_df = fit_and_predict(df, selected_feature, num_days)

                        st.subheader('Forecast Results:')
                        st.dataframe(forecast_df)

                        # Plotting
                        st.subheader('Forecast Plot:')
                        plt.figure(figsize=(10, 5))
                        
                        # Filter the original dataframe to only include the dates within the forecast period
                        forecast_start_date = forecast_df['ds'].min()
                        forecast_end_date = forecast_df['ds'].max()
                        df_filtered = df[(df['Date'] >= forecast_start_date) & (df['Date'] <= forecast_end_date)]
                        
                        # Plot the actual values for the filtered date range
                        plt.plot(df_filtered['Date'], df_filtered[selected_feature], label='Actual')
                        
                        # Plot the forecasted values
                        plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast')
                        plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='gray', alpha=0.2, label='Margin of Error')
                        
                        plt.legend()
                        plt.xlabel('Date')
                        plt.ylabel(selected_feature)
                        plt.title('Actual vs Forecasted Values')
                        st.pyplot(plt)

                    except Exception as e:
                        st.error(f"An error occurred during forecasting: {e}")

        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

if __name__ == '__main__':
    app()
