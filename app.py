from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('lr_model.pickle', 'rb') as f:
    model = pickle.load(f)
with open('scaler_5.pickle', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_by_values():
    """
    Predicts the output based on the form data received from a POST request.
    Expects the form data to contain a dictionary with keys "oil_rate", "depth",
    "water_rate", "tubing_id_1_9992", and "pwh". The values of "oil_rate",
    "depth", "water_rate", and "pwh" are expected to be floating point numbers.
    The value of "tubing_id_1_9992" is expected to be either "Yes" or "No".
    Returns the predicted output as a string rendered by the "home.html" template.
    """
    data = request.form.to_dict()
    for key, value in data.items():
        if key != "tubing_id_1_9992":
            data[key] = float(value)
  

    # Convert the categorical feature 'tubing_id_1_9992' to binary format
    tubing_id_1_9992_binary = 1 if data['tubing_id_1_9992'] == 'Yes' else 0

    # Centering and squaring the input oil rate
    oil_train_mean = 6195.78527607362 # Mean of oil rate in the training set
    oil_c_squared = (data['oil_rate'] - oil_train_mean)**2


    # Create a numpy array with the input values (excluding the categorical feature)
    input_arr = np.array([oil_c_squared, 
                          data['depth'], 
                          data['water_rate'], 
                          tubing_id_1_9992_binary, 
                          data['pwh']])
   

    # Reshape the array to have one row and the same number of columns as the training data
    input_arr = input_arr.reshape(1, -1)

    # # Scale the input using the loaded scaler
    scaled_input = scaler.transform(input_arr)

    # Create a constant array with the same number of rows as the scaled input array
    constant = np.ones((scaled_input.shape[0], 1))

    # # Predict the output for the scaled input using the trained model
    prediction_output = model.predict(np.hstack((constant, scaled_input)))

    return render_template("index.html",
                            prediction_text = str(np.round(prediction_output[0], 2)))
   

if __name__ == "__main__":
    app.run(debug=True)
