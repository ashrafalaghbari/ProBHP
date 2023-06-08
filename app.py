import streamlit as st
from streamlit_shap import st_shap
import pandas as pd
import io
import datetime
import numpy as np
from preprocessing import predict, calculate_shap_online, calculate_shap_batch
import openpyxl
import base64
# Set the page layout
st.set_page_config( page_title = "ProBHP", page_icon = "Active")

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Set page title
st.title("Flowing Bottom Hole Pressure Prediction App")

# Define the instructions page
def instructions_page():
    # Define the text to add to the page
    # Define your CSS style
    st.markdown(
        """
        <style>
        .my-text {
            font-size: 18px;
            font-weight: normal;
            text-align: justify;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



    # Define your Markdown text with the my-text class applied to each paragraph
    text = """
    <p class="my-text">This app is designed to predict the flowing bottom-hole pressure (FBHP) based on the reservoir and well parameters.</p>

    <p class="my-text">Flowing bottomhole pressure (FBHP) is a key parameter in multiphase flow and reservoir engineering.
    It represents the pressure at the bottom of the wellbore where the fluids are flowing out to the surface. 
    FBHP is influenced by several factors such as the reservoir properties, well configuration, fluid properties, and production rate. Accurate prediction of FBHP is crucial for well performance analysis and optimization.</p>

    <p class="my-text">This app uses a feed-forward neural network (FFNN) tuned with Bayesian optimization to predict FBHP based on the reservoir and well parameters. The model was trained using a dataset of 206 samples.</p>

    <p class="my-text">For more information about the model and dataset, please refer to the GitHub repository.
    <a href="https://github.com/ashrafalaghbari/regression_analysis"><img src="https://img.shields.io/badge/GitHub-Repo-lightgrey.svg?logo=github"></a></p>

    <h1 class="my-text"><b>Explainable AI</b></h1>

    <p class="my-text">The app also utilizes Explainable AI techniques, specifically the SHAP (Shapley Additive exPlanations) method, which provides easily interpretable global and local views of model predictions. It offers feature importance for both individual records and batches in a user-friendly manner. This feature allows you to comprehend why the model makes specific predictions, aiding in production optimization.</p>

    <p class="my-text">Deep learning models excel at capturing complex patterns and non-linear relationships. In this context, SHAP proves invaluable for data exploration and analysis, as it is model-agnostic and can be applied to various machine learning and deep learning models. Understanding the decision-making process of the model becomes crucial, and SHAP assists in establishing trust by providing insights into predictions.</p>

    <p class="my-text">However, it's important to note that SHAP has certain limitations. Multicollinearity, which refers to the correlation between predictions, can affect the accuracy of SHAP explanations. Additionally, while SHAP can indicate the contribution of a particular variable towards predictions, it may not account for confounding or proxy variables that influence the relationship between a predictor and the target variable. Therefore, caution should be exercised in interpreting the results. Furthermore, it's essential to remember that SHAP cannot be used for inferring causality and cannot measure the real-world importance of a feature. Understanding these limitations is crucial to drawing accurate conclusions.</p>

    <p class="my-text">If you interested to understand more about SHAP, here are some references that may help you:</p>
    <ul>
    <li class="my-text"><a href="https://www.youtube.com/watch?v=-taOhqkiuIo&t=323s">https://www.youtube.com/watch?v=-taOhqkiuIo&t=323s</a></li>
    <li class="my-text"><a href="https://www.youtube.com/watch?v=b9qqbFudVhI">https://www.youtube.com/watch?v=b9qqbFudVhI</a></li>
    <li class="my-text"><a href="https://christophm.github.io/interpretable-ml-book/shap.html">https://christophm.github.io/interpretable-ml-book/shap.html</a></li>
    </ul>
    """

    # Add the text to the page
    st.markdown(text, unsafe_allow_html=True)

    # # # Apply your CSS style to your Markdown text
    # st.markdown(f'<p class="my-text">{text}</p>', unsafe_allow_html=True)

    st.markdown("# Guidelines for using the app")
    st.write("Please follow the instructions below to ensure accurate predictions when using the app:")
    st.write("Please ensure that you use the exact units as shown in the table "
              "for both online and batch predictions. If you're making batch predictions, make sure "
              "that your CSV file contains all the listed variables in the table to predict the "
              "flowing bottom-hole pressure. Also, ensure that the column names or variable "
              "names in your CSV file match the same style as the `Data Format for Batch Prediction`. "
              "The `Tubing diameter` should have a value of either `4.000`, `3.958`, `3.813`, `2.992`,"
              "`2.441`, or `1.995`, and the data types for the required variables should be numerical. ")
   


    # Read the sample CSV file
    df = pd.read_csv(r'sample.csv')

    # Convert the CSV data to a string
    csv_data = df.to_csv(index=False)

    # Define a function to create a download link for the CSV data
    def download_csv(csv_data, filename):
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download sample CSV file</a>'
        return href

    # Display the sample CSV data and download link
    st.write("Here's an example of a CSV file that can be used for batch prediction:")
    st.dataframe(df)
    st.markdown(download_csv(csv_data, 'sample.csv'), unsafe_allow_html=True)
    st.write("Please note that while the app will produce an output for any input values, "
              "it's important to consider the recommended range for each variable to obtain accurate "
              "and reliable predictions. Using values outside of the recommended range may result in  "
              "less accurate predictions.")
    st.write("")
    st.write("")
       # Define the table
    data = {
        'Variable': ['Oil rate', 'Gas rate', 'Water rate', 'Depth', 'Oil gravity',
                    'Surface temperature', 'Bottom-hole temperature', 'Wellhead pressure',
                    'Tubing diameter'],

        'Units': ['bbl/d', 'Mscf/day', 'bbl/day', 'ft', 'API', '°F', '°F', 'Psi', 'inche'],

        'Range': ['280.000 - 19618.000', '33.600 - 13562.200', '0.000 - 11000.000',
                    '4550.000 - 7100.000', '30.000 - 37.000', '76.000 - 160.000',
                        '157.000 - 215.000', '80.000 - 960.000',
                        '4.000 - 3.958 -3.813 - 2.992 - 2.441 - 1.995' ],

        'Data Format for Batch Prediction': ['oil_rate', 'gas_rate', 'water_rate', 'depth', 'oil_gravity',
                        'stm', 'btm', 'pwh', 'tubing_id']
            }
    
    # Write the table
    st.write("\n")
    st.write("You can use the following table as a reference when making predictions:")  
    df = pd.DataFrame(data)
    df = df  # exclude the first row
    st.table(df)
    st.write("Thank you for using the app!")
    if st.button("Back"):
        st.experimental_rerun()
        st.stop()



def predict_online(data):
    # use the model for inference
    prediction_output = predict(data)
    return prediction_output

def predict_batch(data):
    columns = ['tubing_id_1.995',
                'tubing_id_2.441',
                'tubing_id_2.992',
                'tubing_id_3.813',
                'tubing_id_3.958'
                ]
    tubing_ids_df = pd.DataFrame(data=np.zeros((data.shape[0],len(columns))), columns=columns)
    for i, row in data[['tubing_id']].iterrows():
        column_name = 'tubing_id_' + str(row['tubing_id'])
        if column_name in tubing_ids_df.columns:
            tubing_ids_df.loc[i, column_name] = 1
    data = pd.concat([data, tubing_ids_df], axis=1)
    data.drop(['tubing_id'], axis=1, inplace=True)


    # Perform prediction on the input data
    predictions = []
    for _, input_data in data.iterrows():

        # use the model for inference
        prediction_output = predict(input_data)
        predictions.append(round(prediction_output, 2))

    # Create output dataframe with predicted values
    prediction_df = pd.DataFrame({
        'oil_rate': data['oil_rate'],
        'gas_rate': data['gas_rate'],
        'water_rate': data['water_rate'],
        'depth': data['depth'],
        'oil_gravity': data['oil_gravity'],
        'stm': data['stm'],
        'btm': data['btm'],
        'pwh': data['pwh'],
        'tubing_id_3.958': data['tubing_id_3.958'],
        'tubing_id_3.813': data['tubing_id_3.813'],
        'tubing_id_2.992': data['tubing_id_2.992'],
        'tubing_id_2.441': data['tubing_id_2.441'],
        'tubing_id_1.995': data['tubing_id_1.995'],
        'prediction': predictions
    })

    return prediction_df, data

def run():

    st.sidebar.title("Navigation")
    if st.sidebar.button("Instructions"):
       instructions_page()
    else:
        add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))
     

        if add_selectbox == 'Online':

            oil_rate = st.text_input('Oil Rate', value='')
            gas_rate = st.text_input('Gas Rate', value='')
            water_rate = st.text_input('Water Rate', value='')
            depth = st.text_input('Depth', value='')
            oil_gravity = st.text_input('Oil Gravity', value='')
            stm = st.text_input('STM (Surface Temperature)', value='')
            btm = st.text_input('BTM (Bottom-hole Temperature)', value='')
            pwh = st.text_input('Wellhead Pressure', value='')
            tubing_id_size = st.selectbox('Tubing ID Size', ['3.958', '3.813', '2.992', '2.441', '1.995'],
                                        index=0)
            # Create a dictionary to map the tubing ID sizes to binary values
            tubing_id_values = {'1.995': 0, '2.441': 0, '2.992':0, '3.813': 0, '3.958': 0}
            tubing_id_values[tubing_id_size] = 1

            output=""


            # Create a list of required variables
            required_vars = ['oil_rate', 'gas_rate', 'water_rate', 'depth', 'oil_gravity', 'stm', 'btm', 'pwh',
                            'tubing_id_3.958', 'tubing_id_3.813', 'tubing_id_2.992', 'tubing_id_2.441', 'tubing_id_1.995']
            if st.button("Predict"):

                try:
                # if All variables are present, roceed with prediction
                # Convert input values to floats
                    single_input = [float(val) for val in [oil_rate, gas_rate, 
                                                        water_rate, depth, oil_gravity, stm, btm, pwh]]
                    # Create dictionary of tubing ID values
                    tubing_id_values = {size: int(size == tubing_id_size) 
                                        for size in ['1.995', '2.441', '2.992', '3.813', '3.958']}
                    # Add tubing ID values to input list
                    single_input.extend(list(tubing_id_values.values()))
                    
                    
                    single_output = predict_online(single_input)
                    single_output = round(single_output, 2)
                    single_output = 'The flowing bottom-hole pressure is' + " " + str(single_output) + " " + 'psi'
                    st.success(single_output)
                except:
                    
                    # Create a dictionary to map variable names to their values
                    var_dict = {'Oil Rate': oil_rate, 'Gas Rate': gas_rate, 'Water Rate': water_rate,
                                'Depth': depth, 'Oil Gravity': oil_gravity, 'STM': stm, 'BTM': btm,
                                'Wellhead Pressure': pwh}
                    # st.write([type(input) for input in single_input])
                    # Loop through the dictionary and check each variable
                    for var_name, var_value in var_dict.items():
                        if var_value is not None and var_value.strip() == '':  # Check if the input is empty
                            st.warning(f"Please enter a value for {var_name}.")
                        elif var_value is not None and not var_value.isnumeric():  # Check if the input is not numeric
                            st.warning(f"Please enter a numerical value for {var_name}.")

            if st.button("Explain"):
                
                try:
                    # All variables are present, so proceed with prediction
                # Convert input values to floats
                    single_input = [float(val) for val in [oil_rate, gas_rate, 
                                                        water_rate, depth, oil_gravity, stm, btm, pwh]]

                    # Create dictionary of tubing ID values
                    tubing_id_values = {size: int(size == tubing_id_size) 
                                        for size in ['1.995', '2.441', '2.992', '3.813', '3.958']}
                    # Add tubing ID values to input list
                    single_input.extend(list(tubing_id_values.values()))
                    
                    single_output = predict_online(single_input)
                    single_output = round(single_output, 2)
                    plot_1, _ = calculate_shap_online(single_input, single_output)
                    _, plot_2= calculate_shap_online(single_input, single_output)
                    # add a cetred bold text SHAP Local Interpretation
                    st.markdown("<h1 style='text-align: center; color: black;'>SHAP Local Interpretation</h1>",
                                 unsafe_allow_html=True)
                    st_shap(plot_1, width=800, height=500)
                    st_shap(plot_2, width=800, height=500)

                    # single_output = 'The flowing bottom-hole pressure is' + " " + str(single_output) + " " + 'psi'
                    
                    # st.success(single_output)                  

                except:
                    # Create a dictionary to map variable names to their values
                    var_dict = {'Oil Rate': oil_rate, 'Gas Rate': gas_rate, 'Water Rate': water_rate,
                                'Depth': depth, 'Oil Gravity': oil_gravity, 'STM': stm, 'BTM': btm,
                                'Wellhead Pressure': pwh}

                    # Loop through the dictionary and check each variable
                    for var_name, var_value in var_dict.items():
                        if var_value is not None and var_value.strip() == '':  # Check if the input is empty
                            st.warning(f"Please enter a value for {var_name}.")
                        elif var_value is not None and not var_value.isnumeric():  # Check if the input is not numeric
                            st.warning(f"Please enter a numerical value for {var_name}.")


        if add_selectbox == 'Batch':


            file_upload = st.file_uploader("Upload csv file for predictions", type=["csv", "txt"])
            # Define the required variables and their data types
            required_vars = ['oil_rate', 'gas_rate', 'water_rate', 'depth', 'oil_gravity', 'stm', 'btm', 'pwh', 'tubing_id']
            required_vars = {var: (float, int) for var in required_vars}

            if file_upload is not None:
                if file_upload.name.endswith('.csv'):
                    df = pd.read_csv(file_upload)
                elif file_upload.name.endswith('.txt'):
                    df = pd.read_csv(file_upload, delimiter=' ')
                else:
                    st.error("Invalid file format. Please upload a CSV or TXT file.")
                    return

                # Check if any variable is missing
                missing_vars = [var for var in required_vars if var not in df.columns]
                if missing_vars:
                    st.error(f"The following variable(s) are missing from the CSV file: {', '.join(missing_vars)}"
                             f"\n\nPlease refer to the instructions page.")
                    return
                 
                if not df.applymap(np.isreal).all().all():
                    non_numerical_cols = df.columns[~df.applymap(np.isreal).all(0)]
                    
                    st.error(f"The following columns contain non-numerical values:"
                    f"{', '.join(non_numerical_cols)}. Please ensure all input variables are numerical.")

                # Check if any variable has the wrong data type
         

            # Check if the tubing_id variable has the correct values
                valid_tubing_ids = [3.958, 3.813, 2.992, 2.441, 1.995]
                invalid_tubing_ids = df[~df['tubing_id'].isin(valid_tubing_ids)]
                if not invalid_tubing_ids.empty:
                    st.warning(f"The following rows have an invalid 'tubing_id' value: {invalid_tubing_ids.index.tolist()}. "
                            f"Valid values are: {', '.join(valid_tubing_ids)}.")
                    return

                # If all checks pass, return a success message
                st.success("Data validation passed.")
                
                output_df, input = predict_batch(df)
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
                    col1, col2, col3, col4 = st.columns([1, 3, 3, 1])
                    with col1:
                        if st.button('Explain'):
                            output_df = output_df.iloc[:,-1].values.reshape(-1,1)
                            # Add a centered bold text SHAP Global Interpretation
                            st.markdown("<h1 style='text-align: center; color: black;'>SHAP Global Interpretation</h1>",
                                         unsafe_allow_html=True)
                            plot_1, _ = calculate_shap_batch(input, output_df)
                            _, plot_2 = calculate_shap_batch(input, output_df)
                            st_shap(plot_1, width=900, height=500)

                            st_shap(plot_2, width=1000, height=300)
                    with col2:
                        # Send the bytes to the user as a file download
                        st.download_button(
                            label="Download predictions as CSV file",
                            data= output_bytes_csv,
                            file_name=f"predictions_{timestamp}.csv",
                            mime="text/csv",
                        )
                    with col3:
                        st.download_button(
                            label="Download predictions as Excel file",
                            data=output_bytes_xlsx,
                            file_name=f"predictions_{timestamp}.xlsx",
                            mime="application/vnd.ms-excel",
                        )


if __name__ == '__main__':
    run()