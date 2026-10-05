from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_FILE = DATA_DIR / "clientes_telecom_simulados_raw.csv"
PRE_FILE = DATA_DIR / "clientes_telecom_preprocesados.csv"
METRICS_FILE = DATA_DIR / "metricas_modelos.csv"
SEGMENTS_FILE = DATA_DIR / "segmentos_kmeans.csv"
IMPORTANCE_FILE = DATA_DIR / "importancia_variables.csv"


# -----------------------------------------------------------------------------
# Carga de datos y entrenamiento reproducible del modelo principal del informe
# -----------------------------------------------------------------------------
raw_df = pd.read_csv(RAW_FILE)
df = pd.read_csv(PRE_FILE)
metricas_oficiales = pd.read_csv(METRICS_FILE)
segmentos_oficiales = pd.read_csv(SEGMENTS_FILE)
importancias = pd.read_csv(IMPORTANCE_FILE)

FEATURES = [c for c in df.columns if c not in ["cliente_id", "abandono"]]
X = df[FEATURES].copy()
y = df["abandono"].astype(int)

NUMERIC_FEATURES = X.select_dtypes(include=np.number).columns.tolist()
CATEGORICAL_FEATURES = [c for c in FEATURES if c not in NUMERIC_FEATURES]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            NUMERIC_FEATURES,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            CATEGORICAL_FEATURES,
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

modelo_logistico = Pipeline(
    steps=[
        ("preprocesamiento", preprocessor),
        (
            "modelo",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
                C=1.0,
            ),
        ),
    ]
)
modelo_logistico.fit(X_train, y_train)

y_pred = modelo_logistico.predict(X_test)
y_prob = modelo_logistico.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
metricas_log = {
    "Exactitud": accuracy_score(y_test, y_pred),
    "Precisión": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
}
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Segmentación K-Means descrita en el informe
CLUSTER_FEATURES = [
    "antiguedad_meses",
    "cargo_mensual_usd",
    "tickets_soporte",
    "fallas_servicio_3m",
    "uso_datos_gb",
]
cluster_imputer = SimpleImputer(strategy="median")
cluster_scaler = StandardScaler()
cluster_X_imp = cluster_imputer.fit_transform(df[CLUSTER_FEATURES])
cluster_X_scaled = cluster_scaler.fit_transform(cluster_X_imp)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_raw = kmeans.fit_predict(cluster_X_scaled)

# Se remapea para que el grupo con más fallas corresponda al "segmento 2"
cluster_eval = pd.DataFrame({
    "cluster": cluster_raw,
    "fallas": df["fallas_servicio_3m"].to_numpy(),
})
high_cluster = cluster_eval.groupby("cluster")["fallas"].mean().idxmax()
cluster_map = {high_cluster: 2, 1 - high_cluster: 1}
df_segmentada = df.copy()
df_segmentada["segmento"] = pd.Series(cluster_raw).map(cluster_map).to_numpy()

silhouette_k = {}
for k in range(2, 7):
    labels_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(cluster_X_scaled)
    silhouette_k[k] = silhouette_score(cluster_X_scaled, labels_k)


# -----------------------------------------------------------------------------
# Funciones de visualización
# -----------------------------------------------------------------------------
def fig_distribucion_abandono():
    aux = (
        df["abandono"]
        .map({0: "Permanece", 1: "Abandona"})
        .value_counts()
        .rename_axis("Estado")
        .reset_index(name="Clientes")
    )
    aux["Porcentaje"] = aux["Clientes"] / aux["Clientes"].sum() * 100
    fig = px.bar(
        aux,
        x="Estado",
        y="Clientes",
        text=aux["Porcentaje"].map(lambda x: f"{x:.1f}%"),
        title="Distribución de clientes según abandono",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="Número de clientes", xaxis_title="")
    return fig


def fig_cargo_abandono():
    aux = df.copy()
    aux["Estado"] = aux["abandono"].map({0: "Permanece", 1: "Abandona"})
    fig = px.box(
        aux,
        x="Estado",
        y="cargo_mensual_usd",
        points="outliers",
        title="Cargo mensual según estado de abandono",
        labels={"cargo_mensual_usd": "Cargo mensual (USD)", "Estado": "Estado"},
    )
    return fig


def fig_correlacion():
    cols = [
        "edad",
        "antiguedad_meses",
        "cargo_mensual_usd",
        "tickets_soporte",
        "fallas_servicio_3m",
        "uso_datos_gb",
        "gasto_acumulado_usd",
        "abandono",
    ]
    corr = df[cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        title="Matriz de correlación de variables numéricas",
    )
    return fig


def fig_metricas_modelos():
    long = metricas_oficiales.melt(
        id_vars="Modelo",
        value_vars=["Exactitud", "Precisión", "Recall", "F1", "AUC"],
        var_name="Métrica",
        value_name="Valor",
    )
    fig = px.bar(
        long,
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        range_y=[0, 1],
        title="Comparación de modelos del informe",
    )
    fig.update_layout(xaxis_tickangle=-20)
    return fig


def fig_roc():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Regresión logística (AUC={metricas_log['AUC']:.3f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Clasificador aleatorio", line=dict(dash="dash")
        )
    )
    fig.update_layout(
        title="Curva ROC del modelo principal",
        xaxis_title="Tasa de falsos positivos",
        yaxis_title="Tasa de verdaderos positivos",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return fig


def fig_confusion():
    z = cm
    text = [[f"VN = {z[0,0]}", f"FP = {z[0,1]}"], [f"FN = {z[1,0]}", f"VP = {z[1,1]}"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predice: Permanece", "Predice: Abandona"],
            y=["Real: Permanece", "Real: Abandona"],
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Clientes=%{z}<extra></extra>",
        )
    )
    fig.update_layout(title="Matriz de confusión — Regresión logística")
    return fig


def fig_importancia():
    aux = importancias.sort_values("Importancia", ascending=True)
    fig = px.bar(
        aux,
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Variables más influyentes (importancia por permutación)",
    )
    return fig


def fig_silhouette():
    aux = pd.DataFrame({"k": list(silhouette_k.keys()), "Silhouette": list(silhouette_k.values())})
    fig = px.line(aux, x="k", y="Silhouette", markers=True, title="Selección de k mediante Silhouette")
    fig.update_xaxes(dtick=1)
    return fig


def fig_segmentos():
    aux = df_segmentada.copy()
    aux["Segmento"] = aux["segmento"].map({1: "Segmento 1", 2: "Segmento 2"})
    aux["Estado"] = aux["abandono"].map({0: "Permanece", 1: "Abandona"})
    fig = px.scatter(
        aux,
        x="antiguedad_meses",
        y="fallas_servicio_3m",
        color="Segmento",
        symbol="Estado",
        hover_data=["cliente_id", "cargo_mensual_usd", "tickets_soporte"],
        title="Segmentación K-Means: antigüedad vs. fallas del servicio",
        labels={
            "antiguedad_meses": "Antigüedad (meses)",
            "fallas_servicio_3m": "Fallas del servicio (últimos 3 meses)",
        },
    )
    return fig


# -----------------------------------------------------------------------------
# Funciones interactivas
# -----------------------------------------------------------------------------
def resumen_dataset():
    duplicados = int(raw_df.duplicated(subset=["cliente_id"]).sum())
    faltantes = int(raw_df.isna().sum().sum())
    tasa = df["abandono"].mean() * 100
    cargo = df["cargo_mensual_usd"].mean()
    antig = df["antiguedad_meses"].median()
    return f"""
### Resumen del conjunto de datos

| Indicador | Resultado |
|---|---:|
| Registros iniciales | **{len(raw_df):,}** |
| Clientes únicos después de eliminar duplicados | **{len(df):,}** |
| Duplicados detectados | **{duplicados}** |
| Celdas faltantes en los datos iniciales | **{faltantes}** |
| Tasa de abandono | **{tasa:.1f}%** |
| Cargo mensual promedio | **USD {cargo:.2f}** |
| Mediana de antigüedad | **{antig:.0f} meses** |
"""


def tabla_faltantes():
    faltantes = raw_df.isna().sum().sort_values(ascending=False)
    out = pd.DataFrame({"Variable": faltantes.index, "Valores faltantes": faltantes.values})
    out["Porcentaje"] = (out["Valores faltantes"] / len(raw_df) * 100).round(2)
    return out


def preparar_cliente(
    edad,
    antiguedad_meses,
    cargo_mensual_usd,
    tickets_soporte,
    fallas_servicio_3m,
    uso_datos_gb,
    tipo_contrato,
    metodo_pago,
    tipo_internet,
    gasto_acumulado_usd,
):
    antig = max(float(antiguedad_meses), 0.0)
    cargo = float(cargo_mensual_usd)
    tickets = int(tickets_soporte)
    uso = float(uso_datos_gb)
    row = {
        "edad": int(edad),
        "antiguedad_meses": int(round(antig)),
        "cargo_mensual_usd": cargo,
        "tickets_soporte": tickets,
        "fallas_servicio_3m": int(fallas_servicio_3m),
        "uso_datos_gb": uso,
        "tipo_contrato": tipo_contrato,
        "metodo_pago": metodo_pago,
        "tipo_internet": tipo_internet,
        "gasto_acumulado_usd": float(gasto_acumulado_usd),
        "tickets_por_anio": tickets / (max(antig, 1) / 12.0),
        "costo_promedio_gb": cargo / uso if uso > 0 else np.nan,
        "cliente_nuevo": int(antig <= 6),
    }
    return pd.DataFrame([row], columns=FEATURES)


def predecir_cliente(
    edad,
    antiguedad_meses,
    cargo_mensual_usd,
    tickets_soporte,
    fallas_servicio_3m,
    uso_datos_gb,
    tipo_contrato,
    metodo_pago,
    tipo_internet,
    gasto_acumulado_usd,
    umbral,
):
    nuevo = preparar_cliente(
        edad,
        antiguedad_meses,
        cargo_mensual_usd,
        tickets_soporte,
        fallas_servicio_3m,
        uso_datos_gb,
        tipo_contrato,
        metodo_pago,
        tipo_internet,
        gasto_acumulado_usd,
    )
    prob = float(modelo_logistico.predict_proba(nuevo)[0, 1])
    pred = int(prob >= float(umbral))

    if prob >= 0.70:
        nivel = "ALTO"
    elif prob >= 0.45:
        nivel = "MEDIO"
    else:
        nivel = "BAJO"

    razones = []
    if int(antiguedad_meses) <= 6:
        razones.append("cliente nuevo (≤ 6 meses)")
    if tipo_contrato == "Mensual":
        razones.append("contrato mensual")
    if int(fallas_servicio_3m) >= 2:
        razones.append("varias fallas recientes del servicio")
    if int(tickets_soporte) >= 3:
        razones.append("alta interacción con soporte")
    if not razones:
        razones.append("no presenta varias de las señales operativas destacadas en el informe")

    decision = "RIESGO DE ABANDONO" if pred == 1 else "PROBABLE PERMANENCIA"
    recomendacion = (
        "Priorizar contacto de retención, revisar fallas y experiencia de servicio y ofrecer una alternativa de contrato/beneficio."
        if pred == 1
        else "Mantener monitoreo y seguimiento normal; no se activa una intervención prioritaria con el umbral seleccionado."
    )

    md = f"""
## Resultado de la predicción

**Probabilidad estimada de abandono:** {prob*100:.1f}%  
**Umbral seleccionado:** {float(umbral):.2f}  
**Clasificación:** **{decision}**  
**Nivel de riesgo demostrativo:** **{nivel}**

**Señales observadas:** {', '.join(razones)}.

**Acción orientativa:** {recomendacion}

> El nivel de riesgo y la acción son reglas ilustrativas de la aplicación. El modelo y su preprocesamiento reproducen el enfoque del informe académico con datos simulados.
"""

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Riesgo estimado de abandono"},
            gauge={"axis": {"range": [0, 100]}, "threshold": {"value": float(umbral) * 100}},
        )
    )
    gauge.update_layout(height=300, margin=dict(l=40, r=40, t=60, b=20))
    return md, gauge


def detalle_modelo():
    tn, fp, fn, tp = cm.ravel()
    return f"""
### Modelo principal reproducido en la aplicación

La **regresión logística** se entrena con una división estratificada **80/20**, `random_state=42`, imputación dentro del pipeline, estandarización de variables numéricas, codificación One-Hot y `class_weight='balanced'`.

| Métrica | Valor |
|---|---:|
| Exactitud | **{metricas_log['Exactitud']:.3f}** |
| Precisión | **{metricas_log['Precisión']:.3f}** |
| Recall | **{metricas_log['Recall']:.3f}** |
| F1 | **{metricas_log['F1']:.3f}** |
| AUC | **{metricas_log['AUC']:.3f}** |

La matriz de confusión contiene **{tn} verdaderos negativos**, **{fp} falsos positivos**, **{fn} falsos negativos** y **{tp} verdaderos positivos**. El resultado coincide con la evaluación presentada en el informe para el modelo principal.
"""


def predecir_segmento(antiguedad, cargo, tickets, fallas, uso):
    row = pd.DataFrame(
        [[antiguedad, cargo, tickets, fallas, uso]],
        columns=CLUSTER_FEATURES,
    )
    imp = cluster_imputer.transform(row)
    scaled = cluster_scaler.transform(imp)
    raw_cluster = int(kmeans.predict(scaled)[0])
    segmento = cluster_map[raw_cluster]
    descripcion = (
        "Segmento 2: grupo con mayor promedio de fallas y mayor tasa de abandono en el informe."
        if segmento == 2
        else "Segmento 1: grupo con menor promedio de fallas y menor tasa de abandono en el informe."
    )
    return f"**Segmento asignado: {segmento}.** {descripcion}"


def generar_scoring_completo(umbral):
    X_all = df[FEATURES].copy()
    probs = modelo_logistico.predict_proba(X_all)[:, 1]
    out = df.copy()
    out["probabilidad_abandono"] = probs.round(6)
    out["prediccion_abandono"] = (probs >= float(umbral)).astype(int)
    out["riesgo"] = pd.cut(
        probs,
        bins=[-np.inf, 0.45, 0.70, np.inf],
        labels=["Bajo", "Medio", "Alto"],
    ).astype(str)
    out = out.sort_values("probabilidad_abandono", ascending=False)
    path = OUTPUT_DIR / "clientes_con_riesgo_predicho.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    resumen = pd.DataFrame({
        "Indicador": ["Clientes analizados", "Predichos como abandono", "Riesgo alto (≥70%)", "Riesgo medio (45%-69.9%)"],
        "Valor": [
            len(out),
            int(out["prediccion_abandono"].sum()),
            int((out["riesgo"] == "Alto").sum()),
            int((out["riesgo"] == "Medio").sum()),
        ],
    })
    return resumen, str(path)


# -----------------------------------------------------------------------------
# Interfaz
# -----------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 1280px !important; margin: auto !important;}
.hero {padding: 18px 22px; border-radius: 16px; background: linear-gradient(120deg,#0f172a,#1e3a8a); color:white;}
.hero h1 {margin-bottom: 6px;}
.kpi {font-size: 1.05rem;}
.small-note {font-size: .9rem; opacity: .8;}
"""

with gr.Blocks(title="Minería de Datos — Predicción de abandono") as demo:
    gr.HTML(
        """
        <div class='hero'>
          <h1>Aplicación demostrativa — Minería de Datos</h1>
          <div>Predicción de abandono de clientes en una empresa de telecomunicaciones</div>
          <div class='small-note'>Práctica Experimental 1 · Caso académico simulado · UEA</div>
        </div>
        """
    )

    with gr.Tab("1. Inicio"):
        gr.Markdown(
            """
## Objetivo
Esta aplicación convierte el informe en una demostración interactiva de las fases de **definición del problema, datos, preprocesamiento, modelado, evaluación e interpretación**.

El caso busca anticipar qué clientes presentan mayor probabilidad de abandonar el servicio. Los datos son **simulados**, por lo que la aplicación tiene fines académicos y no debe utilizarse como sistema de decisión real sin validación adicional.
"""
        )
        gr.Markdown(resumen_dataset())
        with gr.Row():
            gr.Plot(fig_distribucion_abandono())
            gr.Plot(fig_cargo_abandono())

    with gr.Tab("2. Datos y exploración"):
        gr.Markdown(
            """
## Exploración del conjunto de datos
El conjunto inicial contiene duplicados y valores faltantes introducidos para demostrar el proceso de limpieza. Después de eliminar duplicados quedan 1.500 clientes únicos.
"""
        )
        with gr.Row():
            gr.Dataframe(raw_df.head(25), label="Muestra de datos iniciales", interactive=False)
            gr.Dataframe(tabla_faltantes(), label="Valores faltantes por variable", interactive=False)
        with gr.Row():
            gr.Plot(fig_correlacion())
            gr.Plot(fig_cargo_abandono())

    with gr.Tab("3. Preprocesamiento"):
        gr.Markdown(
            """
## Flujo de preprocesamiento

1. Eliminación de duplicados por `cliente_id`.
2. Tratamiento de valores extremos del cargo mensual mediante IQR.
3. Imputación de variables numéricas con la mediana y categóricas con la moda dentro del pipeline.
4. Estandarización de variables numéricas.
5. Codificación One-Hot de variables categóricas.
6. Generación de `tickets_por_anio`, `costo_promedio_gb` y `cliente_nuevo`.
7. División estratificada 80% entrenamiento / 20% prueba con `random_state=42`.

**Importante:** el archivo preprocesado conserva algunos valores faltantes porque la imputación se realiza dentro del pipeline del modelo para evitar fuga de información.
"""
        )
        gr.Dataframe(df.head(30), label="Datos después de limpieza y generación de variables", interactive=False)

    with gr.Tab("4. Modelos y evaluación"):
        gr.Markdown(detalle_modelo())
        with gr.Row():
            gr.Dataframe(metricas_oficiales.round(3), label="Métricas comparativas del informe", interactive=False)
            gr.Plot(fig_metricas_modelos())
        with gr.Row():
            gr.Plot(fig_confusion())
            gr.Plot(fig_roc())
        gr.Plot(fig_importancia())
        gr.Markdown(
            """
**Interpretación principal:** la regresión logística fue seleccionada por presentar el AUC más alto del informe y un recall adecuado para detectar clientes en riesgo. KNN obtuvo mayor exactitud, pero un recall muy bajo para la clase abandono, lo que evidencia por qué no debe seleccionarse un modelo únicamente por accuracy.
"""
        )

    with gr.Tab("5. Predictor individual"):
        gr.Markdown(
            """
## Simular un cliente
Modifique los datos y pulse **Calcular riesgo**. La probabilidad proviene del modelo de regresión logística entrenado con el mismo conjunto y enfoque metodológico del informe.
"""
        )
        with gr.Row():
            with gr.Column():
                edad = gr.Slider(18, 85, value=35, step=1, label="Edad")
                antig = gr.Slider(0, 84, value=8, step=1, label="Antigüedad (meses)")
                cargo = gr.Slider(10, 130, value=60, step=0.5, label="Cargo mensual (USD)")
                tickets = gr.Slider(0, 12, value=2, step=1, label="Tickets de soporte")
                fallas = gr.Slider(0, 8, value=2, step=1, label="Fallas del servicio en 3 meses")
            with gr.Column():
                uso = gr.Slider(0, 80, value=20, step=0.5, label="Uso de datos (GB)")
                contrato = gr.Dropdown(["Mensual", "Anual", "Bianual"], value="Mensual", label="Tipo de contrato")
                pago = gr.Dropdown(["Tarjeta", "Transferencia", "Débito automático", "Efectivo"], value="Tarjeta", label="Método de pago")
                internet = gr.Dropdown(["Fibra", "Cable", "Inalámbrico"], value="Fibra", label="Tipo de internet")
                gasto = gr.Number(value=480.0, label="Gasto acumulado (USD)", minimum=0)
                umbral = gr.Slider(0.20, 0.80, value=0.50, step=0.01, label="Umbral de decisión")
        btn = gr.Button("Calcular riesgo", variant="primary")
        with gr.Row():
            resultado = gr.Markdown()
            gauge = gr.Plot()
        btn.click(
            predecir_cliente,
            inputs=[edad, antig, cargo, tickets, fallas, uso, contrato, pago, internet, gasto, umbral],
            outputs=[resultado, gauge],
        )

    with gr.Tab("6. Segmentación K-Means"):
        gr.Markdown(
            f"""
## Segmentación descriptiva
El informe seleccionó **k = 2**. En esta reproducción, el coeficiente Silhouette para k=2 es **{silhouette_k[2]:.3f}**, lo que confirma que los grupos son útiles para perfilar clientes, pero no están fuertemente separados.
"""
        )
        with gr.Row():
            gr.Plot(fig_silhouette())
            gr.Dataframe(segmentos_oficiales.round(3), label="Resumen de segmentos del informe", interactive=False)
        gr.Plot(fig_segmentos())
        gr.Markdown("### Probar asignación de un cliente a un segmento")
        with gr.Row():
            s_antig = gr.Slider(0, 84, value=12, step=1, label="Antigüedad")
            s_cargo = gr.Slider(10, 130, value=55, step=0.5, label="Cargo mensual")
            s_tickets = gr.Slider(0, 12, value=2, step=1, label="Tickets")
            s_fallas = gr.Slider(0, 8, value=3, step=1, label="Fallas")
            s_uso = gr.Slider(0, 80, value=20, step=0.5, label="Uso de datos")
        s_btn = gr.Button("Asignar segmento")
        s_out = gr.Markdown()
        s_btn.click(predecir_segmento, [s_antig, s_cargo, s_tickets, s_fallas, s_uso], s_out)

    with gr.Tab("7. Scoring y exportación"):
        gr.Markdown(
            """
## Generar una lista priorizada de clientes
La aplicación puede calcular la probabilidad de abandono para los 1.500 clientes del conjunto académico y exportar un CSV ordenado desde mayor a menor riesgo. Esto demuestra cómo un modelo podría apoyar una campaña de retención.
"""
        )
        batch_threshold = gr.Slider(0.20, 0.80, value=0.50, step=0.01, label="Umbral para clasificación")
        batch_btn = gr.Button("Generar scoring completo", variant="primary")
        batch_table = gr.Dataframe(label="Resumen del scoring", interactive=False)
        batch_file = gr.File(label="Descargar CSV con probabilidades")
        batch_btn.click(generar_scoring_completo, batch_threshold, [batch_table, batch_file])

    with gr.Tab("8. Conclusiones"):
        gr.Markdown(
            """
## Conclusiones demostradas por la aplicación

- El proceso integra exploración, preprocesamiento, segmentación y clasificación predictiva.
- La tasa de abandono del conjunto simulado es **24,4%**.
- La **regresión logística** reproduce un **AUC ≈ 0,742** y un **recall ≈ 0,671**, por lo que fue seleccionada como modelo principal.
- La matriz de confusión muestra que el modelo detecta **49 de 73** clientes que realmente abandonaron en el conjunto de prueba.
- K-Means identifica un segmento con más fallas del servicio y mayor abandono, aunque el Silhouette es bajo (**≈ 0,172**).
- En un escenario empresarial real sería necesario validar con datos reales, evaluar costos de falsos positivos/falsos negativos, ajustar el umbral y monitorear el modelo periódicamente.
"""
        )

    gr.Markdown(
        """
---
**Nota académica:** los datos son simulados y la aplicación se diseñó para demostrar la metodología de la Práctica Experimental 1 de Minería de Datos.
"""
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
