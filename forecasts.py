import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from io import BytesIO
import xlsxwriter
import numpy as np

def create_features(date):
    """Create time series features based on the date."""
    features = {
        'day_of_week': date.weekday(),
        'month': date.month,
        'day_of_month': date.day,
        'is_weekend': int(date.weekday() in [5, 6])
    }
    
    # Dummy lag features for the purpose of matching model input
    # These values should ideally come from historical data
    features['S_lag_1'] = 0  # Placeholder value
    features['S_lag_2'] = 0  # Placeholder value
    features['S_lag_3'] = 0  # Placeholder value

    return features

def app():
    # Load the model and scaler
    with open('N3B_model_pred.pkl', 'rb') as f:
        model_pred = pickle.load(f)
    with open('N3B_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Create a Streamlit app
    st.title('Prediction App For S')

    # Add explanation
    st.write('This app predicts S based on a given date and provides a 95% confidence interval.')

    # Date input
    selected_date = st.date_input("Select a date")

    # Create a button to trigger the prediction
    if st.button('Make Prediction'):
        try:
            # Create features
            features = create_features(selected_date)

            # Convert to DataFrame
            input_data = pd.DataFrame([features])

            # Scale the input data
            input_scaled = scaler.transform(input_data)

            # Make predictions
            yhat = model_pred.predict(input_scaled)[0]

            # Calculate standard error from residuals
            residuals = model_pred.predict(scaler.transform(X_train)) - y_train
            standard_error = np.std(residuals)

            # Calculate 95% confidence interval
            z_score = 1.96  # For 95% confidence
            yhat_lower = yhat - z_score * standard_error
            yhat_upper = yhat + z_score * standard_error

            # Display the predicted values
            st.subheader('Predicted S:')
            st.write(f"Lower bound (95% CI): {yhat_lower:.2f}")
            st.write(f"Predicted S: {yhat:.2f}")
            st.write(f"Upper bound (95% CI): {yhat_upper:.2f}")

            # Prepare predictions for download
            predictions_df = pd.DataFrame({
                'yhat_lower': [yhat_lower],
                'yhat': [yhat],
                'yhat_upper': [yhat_upper]
            })

            # Function to convert DataFrame to Excel
            def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df.to_excel(writer, index=False, sheet_name='Predictions')
                writer.close()  # Use close instead of save
                processed_data = output.getvalue()
                return processed_data

            # Convert predictions to Excel
            excel_data = to_excel(predictions_df)

            # Download button for Excel
            st.download_button(
                label='Download Predictions as Excel',
                data=excel_data,
                file_name='predicted_s.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Add some styling
    st.markdown("""
    <style>
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    app()
