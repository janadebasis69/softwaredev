from flask import Flask, render_template, send_file
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import plotly.graph_objects as go
from scipy.stats import weibull_min


app = Flask(__name__)

# Load the sensor data
data = pd.read_csv('Datan.csv')
data.columns = data.columns.str.strip()  # Clean column names

# Define thresholds
overcurrent_threshold = 2450
overload_threshold = 603
high_vibration_threshold = 2.5

# Create binary columns for Bayesian network
data['Overcurrent'] = (data['Current'] > overcurrent_threshold).astype(int)
data['Overload'] = (data['Voltage'] > overload_threshold).astype(int)
data['High_Vibration'] = (data['Vibration'] > high_vibration_threshold).astype(int)
data['Failure'] = (data['Failure'] == 'Y').astype(int)

# Calculate probabilities
P_overcurrent = (data['Current'] > overcurrent_threshold).mean()
P_overload = (data['Voltage'] > overload_threshold).mean()
P_high_vibration = (data['Vibration'] > high_vibration_threshold).mean()

# Calculate TTF (Time to Failure) as time between consecutive failures
data['TTF'] = data['HMR'].diff()
data.loc[data['Failure'] == 0, 'TTF'] = np.nan
data['TTF'].fillna(method='ffill', inplace=True)

# Fit a Weibull distribution to TTF values
shape, loc, scale = weibull_min.fit(data['TTF'].dropna(), floc=0)
MTTF = scale * 0.5*(1 + 1/shape)  # Mean Time to Failure using Weibull parameters

# Machine Availability Calculation
MTTR = 5  # Mean Time to Repair (assumed value in hours)
Availability = MTTF / (MTTF + MTTR)
Availability_percentage = round(Availability * 100, 2)

# Define the structure of the Bayesian Network
model = BayesianNetwork([
    ('Overcurrent', 'Failure'),
    ('Overload', 'Failure'),
    ('High_Vibration', 'Failure')
])

# Fit the model using Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Perform inference
inference = VariableElimination(model)

def draw_bayesian_network():
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=14, font_weight='bold', arrowsize=20)
    plt.savefig('static/bayesian_network.png')
    plt.close()

def generate_gauge_plot():
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=Availability_percentage,
        title={'text': "Machine Availability (%)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "blue"}}
    ))
    fig.write_image("static/gauge_plot.png")

def calculate_rul():
    features = data[['Current', 'Voltage', 'Vibration']]
    rul = np.maximum(0, 1000 - np.arange(len(features)))  # Synthetic RUL for demonstration

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, rul, test_size=0.2, random_state=42)

    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    nn_model.compile(optimizer='adam', loss='mse')
    nn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    predicted_rul = nn_model.predict(features_scaled)

    plt.figure(figsize=(10, 6))
    plt.plot(predicted_rul, label='Predicted RUL')
    plt.xlabel('Sample Index')
    plt.ylabel('Remaining Useful Life (RUL)')
    plt.title('RUL Prediction')
    plt.legend()
    plt.savefig('static/rul_graph.png')
    plt.close()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/failure-probability')
def failure_probability():
   # Define a threshold for discretizing probabilities
    threshold = 0.5
    
    # Discretize the continuous probabilities into binary values
    overcurrent_evidence = 1 if P_overcurrent > threshold else 0
    overload_evidence = 1 if P_overload > threshold else 0
    high_vibration_evidence = 1 if P_high_vibration > threshold else 0
    
    # Query the probability of failure given discretized conditions
    prob_failure = inference.query(variables=['Failure'],
                                   evidence={'Overcurrent': overcurrent_evidence, 
                                             'Overload': overload_evidence, 
                                             'High_Vibration': high_vibration_evidence})

    # Extract the failure probability
    failure_prob = prob_failure.values[1]  # Probability of failure
    failure_prob = float(np.float64(failure_prob))

    draw_bayesian_network()
    calculate_rul()

    return render_template('result.html', failure_prob=failure_prob, image_url='static/bayesian_network.png', rul_graph='static/rul_graph.png', gauge_plot='static/gauge_plot.png', MTTF=MTTF)

if __name__ == '__main__':
    app.run(debug=True)
