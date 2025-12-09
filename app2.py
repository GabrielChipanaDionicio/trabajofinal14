import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

# --- 1. Configuración de Página ---
st.set_page_config(page_title="TechPulse Predictor", layout="wide")

# --- 2. Carga de Datos OPTIMIZADA ---
@st.cache_data
def load_sample_data(file_path, n_rows=5000):
    """
    Lee SOLO las primeras 'n_rows' del archivo. 
    Esto hace que la carga sea instantánea sin importar el peso real del archivo.
    """
    # Asegúrate de que las columnas numéricas sean las correctas para tu caso
    try:
        df = pd.read_csv(file_path, nrows=n_rows)
        return df
    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {file_path}")
        return None

# --- 3. Carga del Modelo ---
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"No se encontró el modelo: {model_path}")
        return None

# --- 4. Funciones de Gráficos (Tus funciones originales) ---
def plot_correlation_heatmap(df):
    st.markdown("### Mapa de Calor de Correlaciones")
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Excluir target si existe para ver solo features
    numerical_cols = numerical_cols.drop('Errores_loggeados', errors='ignore') 
    
    if len(numerical_cols) > 0:
        correlation_matrix = df[numerical_cols].corr(method='pearson')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No hay suficientes datos numéricos para el mapa de calor.")

def plot_numerical_box_plots(df):
    st.markdown("### Distribución de Datos (Box Plots)")
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = numerical_cols.drop('Errores_loggeados', errors='ignore')

    num_cols_to_plot = len(numerical_cols)
    if num_cols_to_plot > 0:
        # Ajustamos el grid dinámicamente
        n_cols = 3
        n_rows = (num_cols_to_plot + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.boxplot(y=df[col], ax=axes[i], color="skyblue")
            axes[i].set_title(col, fontsize=10)
            axes[i].set_ylabel('')
        
        # Ocultar ejes vacíos
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        st.pyplot(fig)

# --- 5. Lógica de Predicción ---
def make_prediction(model, input_data_df):
    prediction = model.predict(input_data_df)
    return int(np.round(np.maximum(0, prediction[0])))

# --- 6. Interfaz Principal ---
def main():
    st.title("⚡ TechPulse: Predicción y Análisis")

    # Definir rutas
    data_path = 'techpulse_timeseries_data.csv' # Asegúrate que esté en la misma carpeta o ruta correcta
    model_path = 'best_lgbm_model.joblib'

    # Cargar Modelo
    model = load_model(model_path)

    # ---------------------------------------------------------
    # SECCIÓN 1: PREDICCIÓN (Lo más importante arriba)
    # ---------------------------------------------------------
    st.header("1. Simulador de Predicción")
    st.info("Ajuste los valores para predecir errores en tiempo real.")

    if model:
        col1, col2 = st.columns(2)
        
        with col1:
            tiempo_uso = st.slider('Tiempo uso diario (Horas)', 0.0, 15.0, 5.0)
            consumo_cpu = st.slider('Consumo CPU (%)', 0.0, 100.0, 25.0)
            consumo_memoria = st.slider('Consumo Memoria (GB)', 0.0, 100.0, 10.0)
            trafico_red = st.slider('Tráfico Red (GB)', 0.0, 15.0, 2.0)
        
        with col2:
            temp_promedio = st.slider('Temp. Promedio (°C)', 20.0, 100.0, 60.0)
            temp_maxima = st.slider('Temp. Máxima (°C)', 20.0, 110.0, 70.0)
            almacenamiento = st.slider('Almacenamiento (GB)', 0.0, 800.0, 300.0)

        if st.button('🔍 Predecir Errores', type="primary"):
            input_data = pd.DataFrame([{
                'Tiempo_uso_diario_horas': tiempo_uso,
                'Consumo_CPU_promedio_porcentaje': consumo_cpu,
                'Consumo_memoria_GB': consumo_memoria,
                'Temperatura_promedio_Celsius': temp_promedio,
                'Temperatura_maxima_Celsius': temp_maxima,
                'Espacio_almacenamiento_GB_usado': almacenamiento,
                'Trafico_red_diario_GB': trafico_red
            }])
            
            res = make_prediction(model, input_data)
            st.success(f"Resultados Predichos: **{res} errores**")

    st.markdown("---")

    # ---------------------------------------------------------
    # SECCIÓN 2: GRÁFICOS EDA (Optimizados)
    # ---------------------------------------------------------
    st.header("2. Análisis Exploratorio de Datos (EDA)")
    
    # Usamos un expander para que los gráficos no saturen la vista inicial si no se quieren ver
    with st.expander("Ver Gráficos de Tendencias y Correlaciones (Basado en muestra histórica)", expanded=True):
        
        with st.spinner("Cargando muestra de datos para gráficos..."):
            # AQUÍ ESTÁ EL TRUCO: nrows=5000 carga instantáneo
            df_sample = load_sample_data(data_path, n_rows=5000)
        
        if df_sample is not None:
            # Pestañas para organizar mejor los gráficos
            tab1, tab2 = st.tabs(["🔥 Mapa de Calor", "📦 Distribución (Boxplots)"])
            
            with tab1:
                plot_correlation_heatmap(df_sample)
                st.caption("Nota: Correlaciones calculadas sobre una muestra de 5,000 registros para optimizar velocidad.")
            
            with tab2:
                plot_numerical_box_plots(df_sample)

if __name__ == '__main__':
    main()
