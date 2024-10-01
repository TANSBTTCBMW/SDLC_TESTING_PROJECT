import csv
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.offline import plot
from scipy.optimize import curve_fit

# Ensure Plotly is set to work offline
pio.renderers.default = 'svg'

def read_error_log(csv_file_path):
    error_messages = []
    frequencies = []

    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header

        for row in csv_reader:
            if len(row) == 2:
                error_messages.append(row[0])
                frequencies.append(int(row[1]))

    return error_messages, frequencies

def calculate_probabilities(frequencies):
    total_occurrences = sum(frequencies)
    probabilities = [freq / total_occurrences for freq in frequencies]
    return probabilities

def cocoa_model(x, a, b):
    return a * np.exp(b * x)

def fit_cocoa_model(frequencies):
    x = np.arange(len(frequencies))
    y = np.array(frequencies)

    # Fit the model
    popt, _ = curve_fit(cocoa_model, x, y, p0=(1, 0.1))
    return popt

def predict_using_model(x, params):
    return cocoa_model(x, *params)

def plot_error_probabilities(error_messages, probabilities, frequencies, predicted_probabilities):
    # Create the plot
    fig = go.Figure()

    # Scatter plot for probabilities
    fig.add_trace(go.Scatter(
        x=np.arange(len(error_messages)),  # Use indices for x-axis
        y=probabilities,
        mode='markers+lines',
        marker=dict(size=10, color='blue', opacity=0.7),
        line=dict(width=4),  # Make the line thicker
        hovertemplate="<b>Error: %{text}</b><br>Probability: %{y:.2f}<br>Frequency: %{customdata}",
        text=error_messages,  # Error message for hover info
        customdata=frequencies,  # Frequency data for hover info
        name='Probability'
    ))

    # Add prediction line
    future_x = np.arange(len(frequencies), len(frequencies) + len(predicted_probabilities))
    future_error_messages = [f"Future Error {i+1}" for i in range(len(predicted_probabilities))]
    fig.add_trace(go.Scatter(
        x=future_x,
        y=predicted_probabilities,
        mode='lines',
        line=dict(color='red', dash='dash', width=4),  # Make the prediction line thicker
        name='Predicted Probabilities',
        yaxis='y2'  # Use secondary y-axis
    ))

    # Scatter plot for frequencies on the secondary y-axis
    fig.add_trace(go.Scatter(
        x=np.arange(len(frequencies)),
        y=frequencies,
        mode='lines+markers',
        marker=dict(size=10, color='green', opacity=0.7),
        line=dict(width=2, color='green'),  # Make the line thicker for frequencies
        name='Frequency',
        yaxis='y2'
    ))

    # Update layout with titles, labels, and time representation
    fig.update_layout(
        title="Error Message Probabilities, Frequencies, and Predictions",
        xaxis_title="Time or Error Indices",
        yaxis_title="Probability of Occurrence",
        yaxis2=dict(
            title="Frequency",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=np.concatenate([np.arange(len(error_messages)), future_x]),
            ticktext=np.concatenate([error_messages, future_error_messages])
        ),
        hovermode="closest"
    )

    # Save plot as an HTML file for offline viewing
    plot(fig, filename='error_probabilities_with_frequencies_and_predictions.html', auto_open=True)

if __name__ == '__main__':
    # Path to the CSV file containing error logs
    csv_file_path = '/Users/mannukumar/Downloads/DSA_4th sem/SLDC_TESTING_PROJECT/processed_data.csv'  # Update this with your file path

    # Read the error messages and frequencies from the CSV file
    error_messages, frequencies = read_error_log(csv_file_path)

    # Calculate the probability of each error
    probabilities = calculate_probabilities(frequencies)

    # Fit the Cocoa model and predict future probabilities
    params = fit_cocoa_model(frequencies)
    future_x = np.arange(len(frequencies), len(frequencies) + 10)  # Example future points
    predicted_probabilities = predict_using_model(future_x, params)

    # Plot the error messages and their probabilities along with frequencies and predictions
    plot_error_probabilities(error_messages, probabilities, frequencies, predicted_probabilities)
