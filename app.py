
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

# --- 1. Data Loading Function ---
@st.cache_data
def load_data(file_path, sample_size=None, random_state=42):
    """Loads the dataset from a CSV file and optionally samples it."""
    df = pd.read_csv(file_path)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
    return df

# --- 2. Model Loading Function ---
@st.cache_resource
def load_model(model_path):
    """Loads the trained model from a joblib file."""
    model = joblib.load(model_path)
    return model

# --- 3. Correlation Matrix Heatmap Function ---
def plot_correlation_heatmap(df):
    """Generates and displays a correlation matrix heatmap for the DataFrame."""
    st.subheader("Correlation Matrix Heatmap")
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if 'Errores_loggeados' in numerical_cols:
        # Exclude target if present for heatmap calculation on features only
        numerical_cols = numerical_cols.drop('Errores_loggeados', errors='ignore') 

    correlation_matrix = df[numerical_cols].corr(method='pearson')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Correlation Matrix of Features')
    st.pyplot(fig)

# --- 4. Box Plots for Numerical Features Function ---
def plot_numerical_box_plots(df):
    """Generates and displays box plots for numerical features in the DataFrame."""
    st.subheader("Box Plots of Numerical Features")
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Exclude the target variable 'Errores_loggeados' for outlier visualization, as it's a count variable
    numerical_cols = numerical_cols.drop('Errores_loggeados', errors='ignore')

    num_cols_to_plot = len(numerical_cols)
    if num_cols_to_plot == 0:
        st.write("No numerical columns found to plot box plots.")
        return

    # Determine optimal grid size for subplots
    n_rows = (num_cols_to_plot + 2) // 3 # Roughly 3 columns per row
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() # Flatten for easy iteration

    for i, col in enumerate(numerical_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'Box plot of {col}')
        axes[i].set_ylabel('')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

# --- 5. User Input and Prediction Function ---
def make_prediction(model, input_data_df):
    """Makes a prediction using the loaded model and user input."""
    prediction = model.predict(input_data_df)
    # Ensure prediction is non-negative and integer if target is count data
    return int(np.round(np.maximum(0, prediction[0])))

def prediction_interface(model):
    st.subheader("Predict 'Errores logeados'")

    st.write("Enter feature values for prediction:")

    # Example input widgets (adjust according to your actual features)
    tiempo_uso_diario_horas = st.slider('Tiempo_uso_diario_horas', min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    consumo_cpu_promedio_porcentaje = st.slider('Consumo_CPU_promedio_porcentaje', min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    consumo_memoria_gb = st.slider('Consumo_memoria_GB', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    temperatura_promedio_celsius = st.slider('Temperatura_promedio_Celsius', min_value=20.0, max_value=100.0, value=60.0, step=0.1)
    temperatura_maxima_celsius = st.slider('Temperatura_maxima_Celsius', min_value=20.0, max_value=110.0, value=70.0, step=0.1)
    espacio_almacenamiento_gb_usado = st.slider('Espacio_almacenamiento_GB_usado', min_value=0.0, max_value=800.0, value=300.0, step=0.1)
    trafico_red_diario_gb = st.slider('Trafico_red_diario_GB', min_value=0.0, max_value=15.0, value=2.0, step=0.1)

    input_data = {
        'Tiempo_uso_diario_horas': tiempo_uso_diario_horas,
        'Consumo_CPU_promedio_porcentaje': consumo_cpu_promedio_porcentaje,
        'Consumo_memoria_GB': consumo_memoria_gb,
        'Temperatura_promedio_Celsius': temperatura_promedio_celsius,
        'Temperatura_maxima_Celsius': temperatura_maxima_celsius,
        'Espacio_almacenamiento_GB_usado': espacio_almacenamiento_gb_usado,
        'Trafico_red_diario_GB': trafico_red_diario_gb
    }

    input_df = pd.DataFrame([input_data])

    if st.button('Predict'):
        predicted_errors = make_prediction(model, input_df)
        st.success(f"Predicted 'Errores logeados': {predicted_errors}")

# --- Main Streamlit Application Flow ---
def main():
    st.set_page_config(page_title="TechPulse Anomaly Predictor", layout="wide")
    st.title("TechPulse: Machine Learning for System Anomaly Prediction")
    st.markdown("This application predicts 'Errores logeados' (logged errors) based on system metrics.")

    data_path = '/content/techpulse_timeseries_data.csv'
    model_path = 'best_lgbm_model.joblib'
    sample_size = 10000 # Use the same sample size as in the notebook

    st.subheader("1. Data Loading")
    st.write(f"Loading data from: {data_path} (sampled to {sample_size} records)")
    df_sampled = load_data(data_path, sample_size=sample_size)
    st.write("Sampled data loaded successfully. First 5 rows:")
    st.dataframe(df_sampled.head())

    st.subheader("2. Model Loading")
    st.write(f"Loading trained model from: {model_path}")
    trained_model = load_model(model_path)
    st.write("Model loaded successfully!")

    st.subheader("3. Exploratory Data Analysis (EDA)")
    plot_correlation_heatmap(df_sampled)
    st.markdown(" ") # Add some space
    plot_numerical_box_plots(df_sampled)

    st.subheader("4. Make Predictions")
    prediction_interface(trained_model)

if __name__ == '__main__':
    main()
