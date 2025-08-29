# 🏠 Predicción de precios en Airbnb Madrid

Este proyecto corresponde al Trabajo Fin de Máster (TFM) en **Data Science**, cuyo objetivo fue desarrollar un modelo de *Machine Learning* capaz de predecir los precios de alojamientos de **Airbnb en Madrid**, incorporando factores espaciales, estructurales y de estacionalidad temporal.  

El resultado final es un sistema de predicción implementado en **Python** y desplegado en una interfaz interactiva que permite a los usuarios estimar el precio esperado según características específicas (barrio, tipo de alojamiento, número de habitaciones, baños, etc.) y comparar alternativas.

---

## 📊 Objetivos del proyecto
- Analizar el mercado de Airbnb en Madrid utilizando datos abiertos de [Inside Airbnb](http://insideairbnb.com/).
- Limpiar y preparar los datos eliminando outliers y normalizando variables.
- Comparar diferentes modelos predictivos (**CatBoost, XGBoost, Random Forest**).
- Seleccionar el modelo con mejor rendimiento y explicabilidad.
- Desarrollar una **interfaz amigable** para facilitar la interacción con el modelo.

---

## ⚙️ Metodología
El proyecto se desarrolló siguiendo la metodología **CRISP-DM**, con las fases de:
1. Comprensión del problema.  
2. Obtención y limpieza de datos.  
3. Análisis exploratorio (EDA).  
4. Modelado y evaluación comparativa.  
5. Selección del modelo óptimo.  
6. Desarrollo de la aplicación interactiva.  

---

## 🛠️ Tecnologías empleadas
- **Lenguaje**: Python 3.13.2  
- **Entorno**: Jupyter Notebook  
- **Librerías principales**:
  - pandas, NumPy → análisis y manipulación de datos  
  - matplotlib, seaborn, plotly → visualización de datos  
  - scikit-learn → preprocesamiento, métricas y modelado  
  - xgboost, catboost → gradient boosting  
  - streamlit → creación de la aplicación interactiva  

---

## 📈 Resultados principales
- **Modelo seleccionado**: Random Forest  
- **Métricas obtenidas (conjunto de prueba)**:  
  - MAE: 12.90 €  
  - RMSE: 18.99 €  
  - R²: 0.883  

El modelo demuestra una alta precisión en la predicción de precios, con errores medios muy bajos y una capacidad de explicación del 88% de la variabilidad en los datos.

---

## 🖥️ Interfaz de usuario
La aplicación desarrollada en **Streamlit** permite:  
- Estimar el precio esperado en función de barrio, tipo de alojamiento, habitaciones, baños y mes.  
- Comparar precios entre diferentes meses.  
- Visualizar barrios con precios más cercanos o más baratos que el seleccionado.  
- Explorar un mapa interactivo con la distribución espacial de precios.

### 🖼️ Interfaz principal
![Interfaz de la app](mi_app_airbnb/imagenes/interfaz.png)

### 🗺️ Mapa de barrios
![Mapa de barrios](mi_app_airbnb/imagenes/mapa_barrios.png) 

---

## 📂 Estructura del repositorio

- datos_brutos/               → Datasets originales (Inside Airbnb)
- notebooks/                  → Notebooks de limpieza, análisis y modelado
- mi_app_airbnb/              → Archivos para la app en Streamlit
- 01_TFM_Airbnb_Madrid_anteproyecto.pdf
- 02_TFM_Airbnb_Madrid_diapositivas.pdf
- 03_TFM_Airbnb_Madrid_memorias.pdf
- README.md                   

---

## 👩‍💻 Autora
**Katherine López Ramírez**  
Máster en Data Science – Universidad Europea de Madrid  

---

## 🔗 Enlaces
- 📚 Datos:: [Inside Airbnb](http://insideairbnb.com/get-the-data/)  
- 🌍 Proyecto completo en GitHub: *([este mismo repositorio](https://github.com/kathe-Lopez/TFM-Airbnb-Madrid))*  

---

## 📌 Palabras clave
*Machine Learning, Airbnb, Predicción de precios, Random Forest, Data Science, Streamlit*
