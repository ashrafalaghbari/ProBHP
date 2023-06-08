import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import torch
import torch.nn as nn
import numpy as np
import pickle
import warnings
import shap
warnings.filterwarnings('ignore')

# Load the scaler for input variables
with open('scaler_X.pickle', 'rb') as f:
    scaler_X = pickle.load(f)
# Load the scaler for output variable
with open('scaler_y.pickle', 'rb') as f:
    scaler_y = pickle.load(f)



class FFNN(nn.Module):
    def __init__(self, input_size, n_layers, neurons, activations):
        super(FFNN, self).__init__()
        self.input_size = int(input_size)
        self.n_layers = int(n_layers)
        self.neurons = [int(neuron) for neuron in neurons]
        self.activations = [str(activation) for activation in activations]

        # Define the hidden layers with activations
        self.hidden_layers = nn.Sequential()
        in_features = self.input_size
        for i in range(self.n_layers):
            out_features = self.neurons[i]
            activation = self.get_activation(self.activations[i])
            self.hidden_layers.add_module(f"linear_{i}", nn.Linear(in_features, out_features))
            self.hidden_layers.add_module(f"activation_{i}", activation)
            in_features = out_features

        # Define the output layer
        self.output_layer = nn.Linear(self.neurons[-1], 1)

    def get_activation(self, activation):
        actions = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
        action = actions.get(activation, None)
        if action is None:
            raise ValueError("Invalid activation function: {}".format(activation))
        return action

    def forward(self, x):
        out = self.hidden_layers(x)
        out = self.output_layer(out)
        return out



# Load the model
def load_model():
    # Define the model parameters
    input_size = 13
    n_layers = 3
    neurons = [100, 100, 62]
    activations = ['relu', 'relu', 'sigmoid']

    with torch.no_grad():
        model = FFNN(input_size, n_layers, neurons, activations)
        model.load_state_dict(torch.load('FFNNmodel.pth'))

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        # set the model to evaluation mode
        model.eval()

    return model

def scaling(scaler_X, data):
    # Preprocess the input data
    scaled_data = scaler_X.transform([data])
    return scaled_data
    # return np.array(scaled_data)

# load the trained model
model = load_model()

# set the device to use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(data, scaler_X=scaler_X, scaler_y=scaler_y):
    #Preprocess the input data
    scaled_data = scaling(scaler_X, data)
    #Predict the output
    with torch.no_grad():
        # Convert the preprocessed data to a tensor without changing the memory address
        input_tensor = torch.from_numpy(scaled_data).float().to(device)
        # Predict the output
        output_tensor= model(input_tensor)
        # Scale back the output tensor to the original range
        inversed_output = scaler_y.inverse_transform(output_tensor.cpu().numpy())[0][0]

    return inversed_output


class ShapInput(object):
    """
    Feed the signle input to shap.plots.waterfall(shap_input, max_display=13) as an instance of this class
    since it requires attributes: base_values, values, display_data, feature_names
    """
    def __init__(self, 
                 inversed_base_value,
                 inversed_shap_values, 
                 inversed_feature_values, 
                 feature_names):
        
            self.base_values = inversed_base_value
            self.values = inversed_shap_values
            self.display_data  =  inversed_feature_values
            self.feature_names = feature_names 

def calculate_shap_online(test_X, pred_y, model = model, scaler_y=scaler_y):
    """
    Calculate the shap values for a single instance of input data (SHAP Local Interpretation)
    Parameters:
    ----------
    test_X: a single instance of input data
    pred_y: the model's prediction for the input data
    model: the trained model
    scaler_y: the scaler for the output variable

    return: plot_1, plot_2
    -------
    plot_1: waterfall plot
    plot_2: forece plot
    """

   
    # read train_X_tensor
    train_X_tensor = torch.load('train_X_tensor.pt')
    # Create the DeepExplainer object using the model and the training data tensor
    explainer = shap.DeepExplainer(model, train_X_tensor)
    #Preprocess the input data
    scaled_data = scaling(scaler_X, test_X)
    # Convert the preprocessed data to a tensor without changing the memory address
    test_X_tensor = torch.from_numpy(scaled_data).float().to(device)
    # convert a tensor to a size of 1,13
    test_X_tensor = test_X_tensor.view(1,13)
    # Compute the SHAP values for the test data tensor using the explainer object 
    shap_values = explainer.shap_values(test_X_tensor)
    # inversed base value (average model's predictions for the background samples (train dataset))
    ex = scaler_y.inverse_transform(explainer.expected_value[0].reshape(-1, 1))[0][0]  
    # difference between the base value (expected value or average model's predictions) and instance actual prediction
    contribution = pred_y-ex
    inversed_shap_values = (shap_values/(shap_values).sum())* contribution
    inversed_shap_values =  inversed_shap_values.reshape(13,)
    features_names = ['Oil rate', 'Gas rate', 'Water rate', 'DEPTH', 'Oil gravity', 'STM',
       'BTM', 'Pwh', 'Tubing Id 1.995', 'Tubing Id 2.441', 'Tubing Id 2.992',
       'Tubing Id 3.813', 'Tubing Id 3.958']
    shap_input = ShapInput(ex, inversed_shap_values, 
                       test_X, features_names)
    # SHAP Local Interpretation
    plot_1 = shap.plots.waterfall(shap_input, max_display=13)
    plot_2 = shap.plots.force(ex, inversed_shap_values, feature_names = features_names, features=test_X)

    return plot_1, plot_2

def calculate_shap_batch(test_X, pred_y, model=model, scaler_y=scaler_y):
    """
    Calculate the shap values for a batch of input data (SHAP Global Interpretation)
    Parameters:
    ----------
    test_X: a batch of input data
    pred_y: the model's prediction for the input data
    model: the trained model
    scaler_y: the scaler for the output variable

    return: plot_1, plot_2
    -------
    plot_1: waterfall plot
    plot_2: forece plot
    """
    # read train_X_tensor
    train_X_tensor = torch.load('train_X_tensor.pt')
    # Create the DeepExplainer object using the model and the training data tensor
    explainer = shap.DeepExplainer(model, train_X_tensor)
    #Preprocess the input data
    
    scaled_data = scaler_X.transform(test_X.values)

    # Convert the preprocessed data to a tensor without changing the memory address
    test_X_tensor = torch.from_numpy(scaled_data).float().to(device)
    # Compute the SHAP values for the test data tensor using the explainer object 
    shap_values = explainer.shap_values(test_X_tensor)
    # inversed base value (average model's predictions for the background samples (train dataset))
    ex = scaler_y.inverse_transform(explainer.expected_value[0].reshape(-1, 1))[0][0]  
    # difference between the base value (expected value or average model's predictions) and instance actual prediction
    contribution =  pred_y-ex
    # inversed shap values
    inversed_shap_values  = []
    for i in range(test_X_tensor.shape[0]):
        result = (shap_values[i]/(shap_values[i]).sum())* contribution[i]
        inversed_shap_values.append(result)
    # convert to 2d array
    inversed_shap_values = np.array(inversed_shap_values)
    # difference between the base value and instance actual prediction
    ex = scaler_y.inverse_transform(explainer.expected_value[0].reshape(-1, 1))[0][0]

    # Remove (-) from the features' names
    features_names = ['Oil rate', 'Gas rate', 'Water rate', 'DEPTH', 'Oil gravity', 'STM',
    'BTM', 'Pwh', 'Tubing Id 1.995', 'Tubing Id 2.441', 'Tubing Id 2.992',
    'Tubing Id 3.813', 'Tubing Id 3.958']

    # Plot the SHAP summary plot for the test data (SHAP Global Interpretation)
    plot_1 = shap.summary_plot(inversed_shap_values, test_X, feature_names=features_names, plot_type='bar')
    plot_2 = shap.plots.force(ex, inversed_shap_values, feature_names = features_names, features=test_X)
    return plot_1, plot_2







