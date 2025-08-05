import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from yellowbrick.cluster import KElbowVisualizer
from datetime import datetime
import io

# Configuración de la página
st.set_page_config(
    page_title="Análisis Energético Colombia",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .st-bw {
        background-color: #ffffff;
    }
    .header {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox, .stMultiselect {
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Título principal con logo
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Colombia.svg/1200px-Flag_of_Colombia.svg.png", width=80)
with col2:
    st.title("⚡ Análisis del Balance Energético Colombiano")
    st.markdown("**Herramienta para el análisis exploratorio y predictivo de datos energéticos de Colombia**")

# Función para cargar datos con caché
@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        
        # Limpieza básica de datos
        df = df.dropna(how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Carga de archivos en el sidebar
st.sidebar.header("📂 Carga de Datos")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo (CSV o Excel)",
    type=["csv", "xlsx", "xls"],
    help="Se aceptan archivos CSV o Excel con datos energéticos"
)

# Datos de ejemplo integrados
@st.cache_data
def load_sample_data():
    # Datos ficticios de ejemplo para energía en Colombia
    dates = pd.date_range(start="2010-01-01", end="2022-12-31", freq='M')
    data = {
        'Fecha': dates,
        'Generación_Hidroeléctrica': np.random.normal(5000, 1000, len(dates)),
        'Generación_Térmica': np.random.normal(3000, 800, len(dates)),
        'Generación_Eólica': np.random.normal(200, 50, len(dates)),
        'Generación_Solar': np.random.normal(150, 40, len(dates)),
        'Demanda_Total': np.random.normal(8000, 1500, len(dates)),
        'Precio_Energía': np.random.normal(200, 30, len(dates)),
        'Temperatura': np.random.normal(25, 3, len(dates)),
        'Precipitación': np.random.gamma(5, 2, len(dates)),
        'Exportaciones': np.random.poisson(500, len(dates)),
        'Importaciones': np.random.poisson(300, len(dates))
    }
    df = pd.DataFrame(data)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Año'] = df['Fecha'].dt.year
    df['Mes'] = df['Fecha'].dt.month_name('es')
    return df

use_sample_data = st.sidebar.checkbox("Usar datos de ejemplo", help="Activa esta opción para probar la aplicación con datos de ejemplo")

if use_sample_data:
    df = load_sample_data()
    st.sidebar.success("Datos de ejemplo cargados correctamente!")
elif uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.sidebar.success("Archivo cargado correctamente!")
    else:
        st.sidebar.error("Error al procesar el archivo. Verifica el formato.")
else:
    st.info("👈 Por favor, sube un archivo con datos energéticos o activa la opción de datos de ejemplo")
    st.stop()

# Sidebar con controles
st.sidebar.header("⚙️ Opciones de Análisis")
analysis_type = st.sidebar.selectbox(
    "Seleccione el tipo de análisis",
    ["Exploración de Datos", "Series Temporales", "Correlaciones", "Modelos Predictivos", "Clustering"],
    index=0
)

# Procesamiento inicial de datos
if 'Fecha' in df.columns:
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha')

# Análisis Exploratorio
if analysis_type == "Exploración de Datos":
    st.header("🔍 Exploración Inicial de Datos")
    
    # Mostrar dataframe con opciones
    with st.expander("📊 Visualización de Datos", expanded=True):
        cols = st.columns(3)
        show_raw = cols[0].checkbox("Mostrar datos crudos", value=True)
        show_missing = cols[1].checkbox("Mostrar valores faltantes")
        show_types = cols[2].checkbox("Mostrar tipos de datos")
        
        if show_raw:
            st.dataframe(df.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        if show_missing:
            missing_data = df.isnull().sum().to_frame("Valores Faltantes")
            missing_data["Porcentaje"] = (missing_data["Valores Faltantes"] / len(df)) * 100
            st.dataframe(missing_data.style.background_gradient(cmap='Reds'))
        
        if show_types:
            types_data = pd.DataFrame(df.dtypes, columns=["Tipo de Dato"])
            st.dataframe(types_data.style.background_gradient(cmap='Greens'))
    
    # Estadísticas descriptivas
    with st.expander("📈 Estadísticas Descriptivas", expanded=True):
        st.dataframe(df.describe().T.style.background_gradient(cmap='Purples'))
    
    # Visualizaciones personalizadas
    with st.expander("📉 Visualizaciones Personalizadas", expanded=True):
        col1, col2, col3 = st.columns(3)
        plot_type = col1.selectbox("Tipo de gráfico", ["Histograma", "Boxplot", "Dispersión", "Violín"])
        var_x = col2.selectbox("Variable X", df.columns, index=0)
        
        if plot_type in ["Dispersión", "Boxplot", "Violín"]:
            var_y = col3.selectbox("Variable Y", df.columns, index=min(1, len(df.columns)-1))
        
        if st.button("Generar Gráfico"):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == "Histograma":
                sns.histplot(df[var_x], kde=True, ax=ax, color='skyblue')
                ax.set_title(f'Distribución de {var_x}')
            
            elif plot_type == "Boxplot":
                sns.boxplot(x=df[var_x], y=df[var_y], ax=ax, palette='coolwarm')
                ax.set_title(f'Boxplot de {var_y} por {var_x}')
            
            elif plot_type == "Dispersión":
                sns.scatterplot(x=df[var_x], y=df[var_y], ax=ax, hue=df[var_y], palette='viridis')
                ax.set_title(f'Relación entre {var_x} y {var_y}')
            
            elif plot_type == "Violín":
                sns.violinplot(x=df[var_x], y=df[var_y], ax=ax, palette='magma')
                ax.set_title(f'Distribución de {var_y} por {var_x}')
            
            st.pyplot(fig)

# Análisis de Series Temporales
elif analysis_type == "Series Temporales":
    st.header("⏳ Análisis de Series Temporales")
    
    if 'Fecha' not in df.columns:
        st.warning("No se encontró una columna de fecha en los datos. El análisis temporal requiere una columna 'Fecha'.")
    else:
        # Selección de variables
        time_cols = [c for c in df.columns if c != 'Fecha']
        selected_vars = st.multiselect("Seleccione variables para análisis temporal", time_cols, default=time_cols[:3])
        
        if not selected_vars:
            st.warning("Seleccione al menos una variable para el análisis")
        else:
            # Configuración de gráfico temporal
            st.subheader("Evolución Temporal")
            fig = go.Figure()
            
            for var in selected_vars:
                fig.add_trace(go.Scatter(
                    x=df['Fecha'],
                    y=df[var],
                    name=var,
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
            
            fig.update_layout(
                xaxis_title='Fecha',
                yaxis_title='Valor',
                hovermode='x unified',
                template='plotly_white',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de estacionalidad
            st.subheader("Análisis de Estacionalidad")
            season_var = st.selectbox("Variable para análisis de estacionalidad", selected_vars)
            
            if st.button("Analizar Estacionalidad"):
                df['Mes'] = df['Fecha'].dt.month
                df['Año'] = df['Fecha'].dt.year
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Variación anual
                sns.lineplot(data=df, x='Año', y=season_var, ax=ax1, ci=None)
                ax1.set_title(f'Variación anual de {season_var}')
                ax1.grid(True)
                
                # Variación mensual
                monthly_avg = df.groupby('Mes')[season_var].mean().reset_index()
                sns.barplot(data=monthly_avg, x='Mes', y=season_var, ax=ax2, palette='coolwarm')
                ax2.set_title(f'Variación mensual promedio de {season_var}')
                ax2.grid(True)
                
                st.pyplot(fig)

# Análisis de Correlación
elif analysis_type == "Correlaciones":
    st.header("📊 Análisis de Correlaciones")
    
    # Matriz de correlación
    st.subheader("Matriz de Correlación")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Se necesitan al menos 2 variables numéricas para el análisis de correlación")
    else:
        corr_method = st.selectbox("Método de correlación", ['pearson', 'spearman', 'kendall'])
        corr_matrix = df[numeric_cols].corr(method=corr_method)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            ax=ax
        )
        ax.set_title('Matriz de Correlación')
        st.pyplot(fig)
        
        # Top correlaciones
        st.subheader("Top Correlaciones")
        corr_threshold = st.slider("Umbral de correlación", 0.5, 0.99, 0.7, 0.01)
        
        corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
        corr_pairs = corr_pairs[corr_pairs != 1].drop_duplicates()
        strong_corrs = corr_pairs[abs(corr_pairs) > corr_threshold]
        
        if len(strong_corrs) > 0:
            st.dataframe(strong_corrs.to_frame("Correlación").style.background_gradient(cmap='coolwarm'))
        else:
            st.warning(f"No se encontraron correlaciones superiores a {corr_threshold}")
        
        # Gráfico de dispersión interactivo
        st.subheader("Gráfico de Dispersión Interactivo")
        col1, col2, col3 = st.columns(3)
        x_axis = col1.selectbox("Eje X", numeric_cols, index=0)
        y_axis = col2.selectbox("Eje Y", numeric_cols, index=1)
        color_by = col3.selectbox("Color por", [None] + numeric_cols)
        
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=[df.index],
            trendline="lowess",
            marginal_x="histogram",
            marginal_y="histogram"
        )
        st.plotly_chart(fig, use_container_width=True)

# Modelos Predictivos
elif analysis_type == "Modelos Predictivos":
    st.header("🤖 Modelos Predictivos")
    
    # Selección de características y objetivo
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Se necesitan al menos 2 variables numéricas para modelado predictivo")
    else:
        st.subheader("Configuración del Modelo")
        
        col1, col2 = st.columns(2)
        features = col1.multiselect(
            "Variables predictoras (features)",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )
        target = col2.selectbox("Variable objetivo (target)", numeric_cols)
        
        test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20)
        random_state = st.number_input("Semilla aleatoria", 0, 100, 42)
        
        # Selección de modelos
        st.subheader("Selección de Modelos")
        model_options = {
            "Regresión Lineal": LinearRegression(),
            "Regresión Ridge": Ridge(),
            "Regresión Lasso": Lasso(),
            "Random Forest": RandomForestRegressor(random_state=random_state),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
            "SVR": SVR()
        }
        
        selected_models = st.multiselect(
            "Seleccione modelos a entrenar",
            list(model_options.keys()),
            default=["Regresión Lineal", "Random Forest"]
        )
        
        if st.button("🚀 Entrenar Modelos") and features and target and selected_models:
            try:
                # Preparación de datos
                X = df[features]
                y = df[target]
                
                # Manejo de valores faltantes
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(X)
                y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
                
                # Escalado y división
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size/100, random_state=random_state
                )
                
                # Entrenamiento y evaluación de modelos
                results = []
                models = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Entrenando {model_name}...")
                    progress_bar.progress((i + 1) / len(selected_models))
                    
                    model = model_options[model_name]
                    
                    # Ajuste de hiperparámetros básicos
                    if model_name == "Random Forest":
                        model.set_params(n_estimators=100)
                    elif model_name == "Gradient Boosting":
                        model.set_params(n_estimators=100, learning_rate=0.1)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Métricas
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results.append({
                        'Modelo': model_name,
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAE': mae,
                        'R²': r2
                    })
                    
                    models[model_name] = model
                
                # Mostrar resultados
                results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
                st.subheader("📊 Resultados de los Modelos")
                st.dataframe(results_df.style.background_gradient(cmap='YlGnBu', subset=['R²', 'RMSE']))
                
                # Gráfico de comparación de modelos
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=results_df['Modelo'],
                    y=results_df['R²'],
                    name='R²',
                    marker_color='skyblue'
                ))
                fig.add_trace(go.Bar(
                    x=results_df['Modelo'],
                    y=results_df['RMSE']/results_df['RMSE'].max(),
                    name='RMSE (normalizado)',
                    marker_color='salmon'
                ))
                fig.update_layout(
                    barmode='group',
                    title='Comparación de Modelos',
                    yaxis_title='Valor',
                    xaxis_title='Modelo'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Visualización de predicciones vs reales
                st.subheader("🔮 Predicciones vs Valores Reales")
                best_model_name = results_df.iloc[0]['Modelo']
                best_model = models[best_model_name]
                y_pred_best = best_model.predict(X_test)
                
                fig = px.scatter(
                    x=y_test,
                    y=y_pred_best,
                    labels={'x': 'Valor Real', 'y': 'Predicción'},
                    trendline="lowess",
                    title=f"Predicciones vs Reales - {best_model_name}"
                )
                fig.add_shape(
                    type="line",
                    x0=min(y_test),
                    y0=min(y_test),
                    x1=max(y_test),
                    y1=max(y_test),
                    line=dict(color="Red", width=2, dash="dot")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Importancia de características (si aplica)
                if hasattr(best_model, 'feature_importances_'):
                    st.subheader("📌 Importancia de Características")
                    importances = pd.DataFrame({
                        'Variable': features,
                        'Importancia': best_model.feature_importances_
                    }).sort_values('Importancia', ascending=False)
                    
                    fig = px.bar(
                        importances,
                        x='Importancia',
                        y='Variable',
                        orientation='h',
                        title='Importancia de Variables'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Descarga del modelo
                st.subheader("💾 Exportar Modelo")
                if st.button("Descargar Mejor Modelo"):
                    import joblib
                    from io import BytesIO
                    
                    buffer = BytesIO()
                    joblib.dump(best_model, buffer)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="Descargar modelo",
                        data=buffer,
                        file_name=f"modelo_{best_model_name.replace(' ', '_').lower()}.pkl",
                        mime="application/octet-stream"
                    )
            
            except Exception as e:
                st.error(f"Error al entrenar modelos: {str(e)}")

# Análisis de Clustering
else:
    st.header("👥 Análisis de Clustering")
    
    # Selección de variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Se necesitan al menos 2 variables numéricas para clustering")
    else:
        cluster_features = st.multiselect(
            "Variables para clustering",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )
        
        if not cluster_features:
            st.warning("Seleccione al menos 2 variables para clustering")
        else:
            # Preprocesamiento
            X = df[cluster_features]
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Selección de algoritmo
            algorithm = st.selectbox("Algoritmo de Clustering", ["K-Means", "DBSCAN"])
            
            if algorithm == "K-Means":
                st.subheader("K-Means Clustering")
                
                # Método para determinar k
                k_method = st.radio("Método para determinar k", ["Manual", "Elbow Method"], index=1)
                
                if k_method == "Manual":
                    n_clusters = st.slider("Número de clusters (k)", 2, 10, 3)
                else:
                    st.info("Calculando método del codo...")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    visualizer = KElbowVisualizer(KMeans(random_state=42), k=(2,10), ax=ax)
                    visualizer.fit(X_scaled)
                    visualizer.finalize()
                    st.pyplot(fig)
                    n_clusters = visualizer.elbow_value_
                    st.success(f"Número óptimo de clusters sugerido: {n_clusters}")
                
                # Ejecutar K-Means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                
            else:  # DBSCAN
                st.subheader("DBSCAN Clustering")
                col1, col2 = st.columns(2)
                eps = col1.slider("EPS (distancia)", 0.1, 2.0, 0.5, 0.1)
                min_samples = col2.slider("Mínimo de muestras", 1, 20, 5)
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(X_scaled)
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                st.info(f"DBSCAN identificó {n_clusters} clusters y {sum(clusters == -1)} puntos como ruido")
            
            # Visualización de resultados
            if n_clusters > 0:
                st.subheader("Visualización de Clusters")
                
                # Reducción de dimensionalidad para visualización
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(X_scaled)
                
                # Crear dataframe para visualización
                cluster_df = pd.DataFrame({
                    'PC1': principal_components[:, 0],
                    'PC2': principal_components[:, 1],
                    'Cluster': clusters,
                    'Size': 50
                })
                
                # Gráfico interactivo
                fig = px.scatter(
                    cluster_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    size='Size',
                    hover_data={'PC1': False, 'PC2': False, 'Cluster': True, 'Size': False},
                    title='Visualización de Clusters (PCA)',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estadísticas por cluster
                st.subheader("Características por Cluster")
                df_cluster = df.copy()
                df_cluster['Cluster'] = clusters
                
                # Resumen estadístico
                cluster_stats = df_cluster.groupby('Cluster')[cluster_features].mean().T
                st.dataframe(cluster_stats.style.background_gradient(cmap='Blues', axis=1))
                
                # Gráfico de radar por cluster
                if n_clusters > 1:
                    st.subheader("Perfil de Clusters (Radar Chart)")
                    
                    # Normalizar datos para radar chart
                    scaler = MinMaxScaler()
                    radar_data = scaler.fit_transform(cluster_stats.T)
                    radar_data = pd.DataFrame(radar_data, columns=cluster_stats.index)
                    radar_data['Cluster'] = cluster_stats.columns
                    
                    fig = go.Figure()
                    
                    for i, row in radar_data.iterrows():
                        fig.add_trace(go.Scatterpolar(
                            r=row.values[:-1],
                            theta=cluster_stats.index,
                            fill='toself',
                            name=f'Cluster {int(row["Cluster"])}'
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True,
                        title="Comparación de Clusters (Normalizado)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No se encontraron clusters válidos")

# Notas finales
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **🔍 Herramienta de análisis energético**\n
    Creada con Streamlit | Python | Scikit-learn\n
    Datos de ejemplo: Generación ficticia basada en patrones energéticos de Colombia\n
    [Sugerencias o reporte de errores](mailto:analisis.energetico@example.com)
    """
)

# Pie de página
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.9em;">
    <p>© 2023 Análisis Energético Colombia | Versión 1.0.0</p>
    <p>Esta herramienta es para fines educativos y demostrativos</p>
    </div>
    """,
    unsafe_allow_html=True
)
