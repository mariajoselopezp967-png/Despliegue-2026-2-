import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("""
Esta aplicación predice el resultado del curso utilizando un modelo Bagging Optimizado,
procesando la entrada del usuario con los transformadores cargados.
""")

# 1. Cargar artefactos necesarios
try:
    columnas_one_hot = joblib.load('/content/one_hot_columns.joblib')
    scaler = joblib.load('/content/min_max_scaler.joblib')
    modelo_bagging = joblib.load('/content/bagging_optimizado.joblib')
    st.success("Modelos y transformadores cargados exitosamente.")
except Exception as e:
    st.error(f"Error al cargar los archivos .joblib: {e}")

# 2. Formulario de entrada de usuario
st.header("Datos del Estudiante")

# Obtener categorías únicas de Felder
estilos_felder = ['equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal', 'visual']

felder_input = st.selectbox("Estilo de aprendizaje (Felder):", estilos_felder)
examen_input = st.number_input("Nota del Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.8, step=0.1)

if st.button("Realizar Predicción"):
    # 3. Crear DataFrame inicial de una fila
    df_usuario = pd.DataFrame([{
        'Felder': felder_input,
        'Examen_admisión': examen_input
    }])
    
    # 4. Procesamiento de los datos de entrada
    df_procesado = df_usuario.copy()
    
    # Aplicar One-Hot manualmente basándonos en la lista cargada
    if 'Felder' in df_procesado.columns:
        valor_felder = df_procesado['Felder'].iloc[0]
        for col in columnas_one_hot:
            if col.startswith('Felder_'):
                categoria = col.replace('Felder_', '')
                df_procesado[col] = 1.0 if valor_felder == categoria else 0.0
        df_procesado = df_procesado.drop(columns=['Felder'])
    
    # Normalizar la variable 'Examen_admisión'
    if 'Examen_admisión' in df_procesado.columns:
        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])[0][0]
        df_procesado = df_procesado.drop(columns=['Examen_admisión'])
    
    # Reordenar las columnas según la estructura esperada
    columnas_finales = [col for col in columnas_one_hot if col in df_procesado.columns]
    df_procesado = df_procesado[columnas_finales]
    
    # Mostrar el registro procesado para control
    st.subheader("Registro Procesado")
    st.dataframe(df_procesado)
    
    # 5. Predicción
    try:
        prediccion = modelo_bagging.predict(df_procesado)
        st.subheader("Resultado de la Predicción")
        st.metric(label="Valor Predicho (Nota final esperada)", value=f"{prediccion[0]:.2f}")
        
        # Intentar predecir probabilidades si es clasificación
        try:
            probabilidades = modelo_bagging.predict_proba(df_procesado)
            st.write("Probabilidades:", probabilidades[0])
        except AttributeError:
            pass
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
