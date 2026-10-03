import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza una predicción utilizando un modelo de Bagging pre-entrenado.")

# Función para procesar y predecir sobre un DataFrame
def procesar_y_predecir(df_input):
    try:
        df_procesado = df_input.copy()

        # 1. Eliminar variables innecesarias si existen
        columnas_a_eliminar = ['ID', 'Año - Semestre', 'Nota_final', 'Aprobo']
        df_procesado = df_procesado.drop(columns=[col for col in columnas_a_eliminar if col in df_procesado.columns], errors='ignore')

        # 2. Cargar y aplicar el transformador/columnas de One-Hot para la variable Felder
        one_hot_transformer = joblib.load('one_hot_columns.joblib')

        if isinstance(one_hot_transformer, list):
            si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
            for col_name in si_columnas_one_hot:
                valor_esperado = col_name.replace('Felder_', '')
                df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(int)
        else:
            df_encoded = pd.get_dummies(df_procesado[['Felder']])
            df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
            si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]

        # Eliminar la variable original Felder
        df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')

        # Asegurar que existan todas las columnas que el modelo espera
        if isinstance(one_hot_transformer, list):
            for col in si_columnas_one_hot:
                if col not in df_procesado.columns:
                    df_procesado[col] = 0

        # 3. Normalizar la variable Examen_admisión con 'min_max_scaler.joblib'
        scaler = joblib.load('min_max_scaler.joblib')
        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
        df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')

        # Reordenar las columnas conforme lo espera el modelo
        columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
        df_procesado = df_procesado[columnas_ordenadas]

        # 4. Predicción con 'bagging_optimizado.joblib'
        model = joblib.load('bagging_optimizado.joblib')
        predicciones = model.predict(df_procesado)

        return df_procesado, predicciones
    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento o la predicción: {e}")
        return None, None

# Selector del método de entrada
metodo_entrada = st.radio("Selecciona el método de entrada de datos:", ("Entrada Manual", "Subir Archivo Excel"))

if metodo_entrada == "Entrada Manual":
    st.header("Datos de Entrada Manual")
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder)
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)

    if st.button("Realizar Predicción"):
        df_input = pd.DataFrame({'Felder': [felder_input], 'Examen_admisión': [examen_input]})
        df_proc, preds = procesar_y_predecir(df_input)
        if preds is not None:
            st.subheader("Datos Procesados para el Modelo")
            st.dataframe(df_proc)
            st.success(f"La predicción del modelo (Nota Final Estimada) es: {preds[0]:.4f}")

else:
    st.header("Subir Archivo Excel")
    st.write("El archivo Excel debe contener al menos las columnas `Felder` y `Examen_admisión`.")
    archivo_cargado = st.file_uploader("Elige un archivo Excel", type=["xlsx"])

    if archivo_cargado is not None:
        try:
            df_excel = pd.read_excel(archivo_cargado)
            st.subheader("Vista previa de los datos subidos")
            st.dataframe(df_excel.head())

            if st.button("Procesar y Predecir Lote"):
                if 'Felder' in df_excel.columns and 'Examen_admisión' in df_excel.columns:
                    df_proc, preds = procesar_y_predecir(df_excel)
                    if preds is not None:
                        df_resultado = df_excel.copy()
                        df_resultado['Prediccion_Nota_Final'] = preds
                        st.subheader("Resultados de las Predicciones")
                        st.dataframe(df_resultado)
                else:
                    st.error("El archivo Excel debe tener las columnas 'Felder' y 'Examen_admisión'.")
        except Exception as e:
            st.error(f"Error al leer el archivo Excel: {e}")
