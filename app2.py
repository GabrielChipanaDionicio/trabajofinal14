import pandas as pd
import streamlit as st
import joblib
import numpy as np

# --- 1. Configuración de la Página ---
st.set_page_config(page_title="TechPulse Predictor", layout="centered")

# --- 2. Cargar solo el Modelo ---
# Usamos cache_resource para que el modelo se cargue una sola vez en memoria
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"No se encontró el archivo del modelo en: {model_path}")
        return None

# --- 3. Función de Predicción ---
def make_prediction(model, input_data_df):
    prediction = model.predict(input_data_df)
    # Asegurar que sea un entero positivo
    return int(np.round(np.maximum(0, prediction[0])))

# --- 4. Interfaz Principal ---
def main():
    st.title("⚡ TechPulse: Predicción de Fallas en Tiempo Real")
    st.markdown("Ingrese las métricas del sistema para predecir 'Errores logeados'.")

    # Ruta del modelo (Asegúrate que este archivo esté en la misma carpeta que tu script)
    model_path = 'best_lgbm_model.joblib'
    
    # Cargamos el modelo
    model = load_model(model_path)

    if model is not None:
        st.divider() # Línea visual divisoria
        
        # --- Columnas para mejor organización visual ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Métricas de Uso")
            tiempo_uso = st.slider('Tiempo de uso diario (Horas)', 0.0, 15.0, 5.0, 0.1)
            consumo_cpu = st.slider('Consumo CPU Promedio (%)', 0.0, 100.0, 25.0, 0.1)
            consumo_memoria = st.slider('Consumo Memoria (GB)', 0.0, 100.0, 10.0, 0.1)
            trafico_red = st.slider('Tráfico de Red Diario (GB)', 0.0, 15.0, 2.0, 0.1)

        with col2:
            st.subheader("Métricas de Estado")
            temp_promedio = st.slider('Temp. Promedio (°C)', 20.0, 100.0, 60.0, 0.1)
            temp_maxima = st.slider('Temp. Máxima (°C)', 20.0, 110.0, 70.0, 0.1)
            almacenamiento = st.slider('Almacenamiento Usado (GB)', 0.0, 800.0, 300.0, 0.1)

        # Preparar datos para el modelo (deben tener los mismos nombres que usaste al entrenar)
        input_data = {
            'Tiempo_uso_diario_horas': tiempo_uso,
            'Consumo_CPU_promedio_porcentaje': consumo_cpu,
            'Consumo_memoria_GB': consumo_memoria,
            'Temperatura_promedio_Celsius': temp_promedio,
            'Temperatura_maxima_Celsius': temp_maxima,
            'Espacio_almacenamiento_GB_usado': almacenamiento,
            'Trafico_red_diario_GB': trafico_red
        }

        input_df = pd.DataFrame([input_data])

        st.markdown("---")
        
        # Botón de predicción centrado y grande
        if st.button('🔍 Ejecutar Predicción', use_container_width=True):
            
            with st.spinner('Calculando probabilidad de errores...'):
                predicted_errors = make_prediction(model, input_df)
            
            # Mostrar resultado como una Métrica Grande (KPI)
            st.success("Cálculo finalizado")
            
            # Diseño de métrica visual
            col_res1, col_res2, col_res3 = st.columns([1,2,1])
            with col_res2:
                st.metric(
                    label="Errores Logeados Estimados", 
                    value=predicted_errors, 
                    delta="Alerta de Riesgo" if predicted_errors > 5 else "Estable",
                    delta_color="inverse"
                )

if __name__ == '__main__':
    main()
