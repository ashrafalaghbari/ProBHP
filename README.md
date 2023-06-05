# ProFBHP: Intelligent FBHP Estimation and Analysis
[![Made with Python](https://img.shields.io/badge/Made%20with-Python%203.10.7-blue.svg)](https://www.python.org/)
## Description
ProBHP is an AI-powered system for accurate flowing bottom hole pressure (FBHP) estimation and analysis in oil wells. It utilizes advanced machine learning and Explainable AI techniques to optimize production, monitor fluid movements, and assess reservoir performance. ProBHP provides cost-effective solutions for FBHP estimation in single-phase and multiphase flow scenarios, overcoming the limitations of conventional methods. With superior accuracy and real-time monitoring, ProBHP enhances wellbore management and production optimization.

# Problem Statement & Motivation

The estimation of flowing bottom hole pressure (FBHP) in oil wells is crucial for monitoring fluid movements, optimizing production, quantifying reservoir performance, and understanding wellbore behavior. However, conventional methods for estimating FBHP are expensive and unreliable, especially when dealing with multiphase flow. Existing empirical correlations and mechanistic models developed in laboratory settings often yield inaccurate results when applied in the field. Additionally, obtaining FBHP using pre-installed permanent gauges in smart wells and well-testing analysis is costly and time-consuming, requiring constant calibration and maintenance. Therefore, there is a clear need for a more efficient and accurate approach.

# Data Collection:

To address the FBHP estimation problem, a dataset consisting of 206 data points was collected from various sources, including Govier and Fogarasi (1975) and Asheim (1986). These sources conducted BHP surveys by deploying down-hole pressure gauges just above the perforations to record the FBHP. The collected data serves as a valuable resource for developing and validating improved FBHP estimation methods.

# Model Development and Deployment

In this project, I have developed a model capable of accurately predicting FBHP using a Feedforward Neural Network (FFNN). To optimize the FFNN and obtain the optimal hyperparameters, I employed Bayesian optimization, which proved to be highly efficient in implementation. This approach allowed me to fine-tune the network and achieve superior performance in FBHP estimation.

The developed model has been successfully deployed and is capable of performing both online and batch predictions. This means that it can provide real-time FBHP estimates during ongoing operations, as well as handle large datasets for retrospective analysis. The deployment of the model enables continuous monitoring of FBHP and enhances decision-making in oil and gas well operations.

![fbhp](https://github.com/ashrafalaghbari/ProBHP/assets/98224412/19523b40-765c-404c-a3df-7c2b4fe677e8)


# Explainable AI and Feature Importance:

To enhance the interpretability of the FBHP prediction model, I incorporated Explainable AI techniques using SHAP (SHapley Additive exPlanations) values. SHAP values provide insight into the contribution of each feature to the model's predictions. By understanding the feature importance, I gain valuable insights into why the model is making specific predictions.

The utilization of SHAP values allows me to interpret the results of the FBHP model accurately. It provides a transparent and understandable framework to comprehend the factors influencing FBHP and facilitates better decision-making in well operations.

This app allows you to input relevant parameters and receive accurate FBHP predictions in real-time or for batch analysis, depending on your requirements. The integration of Explainable AI ensures that you can also access feature importance information, gaining valuable insights into the model's decision-making process and enhancing your understanding of FBHP dynamics in oil wells.

![wkflow_bhp](https://github.com/ashrafalaghbari/ProBHP/assets/98224412/b15ea5c6-3c71-433b-9c07-f345021ad99a)


# Installation

Follow these steps to install and run the project locally:

Set up a virtual environment (optional but recommended):

```bash
python -m venv env
env\Scripts\activate.bat
```

```bash
git clone https://github.com/ashrafalaghbari/ProBHP.git
cd project-directory
pip install -r requirements.txt
streamlit run app.py
```

Access the web application by opening the following URL in your web browser:

```bash
http://localhost:8501
```
Follow the instructions on the web application to use the project.

If you prefer to use a Docker image, you can follow these additional steps:

Pull the Docker image from Docker Hub:
```bash
docker pull fbhpapp:0.1
```
Run the Docker container:
```bash
docker run -p 8501:8501 fbhpapp:0.1
```
Access the web application using the same URL as mentioned above.

# References
* [An Automated Flowing Bottom-Hole Pressure Prediction for a Vertical Well Having Multiphase Flow UsingComputational Intelligence Techniques](https://onepetro.org/SPESATS/proceedings-abstract/18SATS/All-18SATS/215548)

* [Prediction of pressure in different two-phase flow conditions: Machine learning applications](https://www.sciencedirect.com/science/article/abs/pii/S0263224120311775)

* [A new flowing bottom hole pressure prediction model using M5 prime decision tree approach](https://link.springer.com/article/10.1007/s40808-021-01211-7)

* [Real-time prognosis of flowing bottom-hole pressure in a vertical well for a multiphase flow using computational intelligence techniques](https://link.springer.com/article/10.1007/s13202-019-0728-4)

* [Machine learning models to predict bottom hole pressure in multi-phase flow in vertical oil production wells](https://onlinelibrary.wiley.com/doi/abs/10.1002/cjce.23526)

* [An Intelligent Solution to Forecast Pressure Drop in a Vertical Well Having Multiphase Flow Using Functional Network Technique](https://onepetro.org/SPEPATS/proceedings-abstract/18PATC/All-18PATC/215853)

# License

[MIT](https://github.com/ashrafalaghbari/ProBHP/blob/main/LICENSE)


# Contact

If you have any questions or encounter any issues running this project, please feel free to [open an issue](https://github.com/ashrafalaghbari/ProBHP/issues) or contact me directly at [ashrafalaghbari@hotmail.com](mailto:ashrafalaghbari@hotmail.com). I'll be happy to help!


