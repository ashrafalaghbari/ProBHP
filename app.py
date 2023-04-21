import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import xlsxwriter
import datetime


# Load the trained model and scaler
with open('lr_model.pickle', 'rb') as f:
    model = pickle.load(f)
with open('scaler_5.pickle', 'rb') as f:
    scaler = pickle.load(f)

def preprocess(scaler, data):
    tubing_id_1_995_binary = 1 if data['tubing_id_1_995'] == 'Yes' else 0
    mean_oil_train = 6195.78527607362 # Mean of Oil_rate in the training set
    oil_c_squared = (data['oil_rate'] - mean_oil_train)**2
    input_arr = [oil_c_squared, data["depth"], data["water_rate"], tubing_id_1_995_binary, data["pwh"]]
    scaled_input = scaler.transform([input_arr])
    constant = [1]
    input_with_constant = constant + list(scaled_input[0])
    
    return input_with_constant


def predict_online(model, scaler, data):
    preprocessed_data = preprocess(scaler, data)
    prediction_output = model.predict([preprocessed_data])[0]
    return prediction_output

def predict_batch(model, scaler, data):

    # Perform prediction on the input data
    predictions = []
    for _, input_data in data.iterrows():
        preprocessed_data = preprocess(scaler, input_data)
        prediction_output = model.predict([preprocessed_data])[0]
        predictions.append(round(prediction_output, 2))

    # Create output dataframe with predicted values
    prediction_df = pd.DataFrame({
        'oil_rate': data['oil_rate'],
        'depth': data['depth'],
        'water_rate': data['water_rate'],
        'tubing_id_1_995': data['tubing_id_1_995'],
        'pwh': data['pwh'],
        'prediction': predictions
    })

    return prediction_df

def run():

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.title("Flowing Bottom Hole Pressure Prediction App")

    if add_selectbox == 'Online':

        oil_rate = st.text_input('Oil Rate', value='')
        depth = st.text_input('Depth', value='')
        water_rate = st.text_input('Water Rate', value='')
        tubing_id_1_995 = st.selectbox('Tubing ID 1.995', ('Yes', 'No'))
        pwh = st.text_input('Wellhead Pressure', value='')

        output=""

        if st.button("Predict"):
            # Check if any variable is missing
            missing_vars = []
            if not oil_rate:
                missing_vars.append('Oil Rate')
            if not depth:
                missing_vars.append('Depth')
            if not water_rate:
                missing_vars.append('Water Rate')
            if not pwh:
                missing_vars.append('Wellhead Pressure')

            # If any variable is missing, display an error message
            if missing_vars:
                st.warning("Please enter a value for the following variables: {}".format(", ".join(missing_vars)))

            else:
                # All variables are present, so proceed with prediction
                input_dict = {'oil_rate': float(oil_rate),
                                'depth': float(depth), 
                                'water_rate': float(water_rate), 
                                'tubing_id_1_995': tubing_id_1_995, 
                                'pwh': float(pwh)}
                output = predict_online(model, scaler, input_dict)
                output = round(output, 2)
                output = 'The flowing bottom-hole pressure is' + " " + str(output) + " " + 'psi'

        st.success(output)

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            df = pd.read_csv(file_upload)
            output_df = predict_batch(model, scaler, df)
            st.write(output_df)

            # Display the download button only if predictions are available
            if not output_df.empty:
                # Set the filename based on the current date and time
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_bytes_csv = output_df.to_csv(index=False).encode()
                # Convert the DataFrame to an Excel file in bytes
                # Write files to in-memory strings using BytesIO
                output = io.BytesIO()
                # Write the data to the Excel file using pandas
                output_df.to_excel(output, index=False)
                output_bytes_xlsx = output.getvalue()
                # Send the bytes to the user as a file download
                st.download_button(
                    label="Download predictions as CSV file",
                    data= output_bytes_csv,
                    file_name=f"predictions_{timestamp}.csv",
                    mime="text/csv",
                )
                st.download_button(
                    label="Download predictions as Excel file",
                    data=output_bytes_xlsx,
                    file_name=f"predictions_{timestamp}.xlsx",
                    mime="application/vnd.ms-excel",
                )


if __name__ == '__main__':
    run()