import gradio as gr
import numpy as np
import pandas as pd
from skops.io import get_untrusted_types, load
import shap
import matplotlib
import matplotlib.pyplot as plt
import warnings

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Cargar modelo ──────────────────────────────────────────────────────────────

unsafe = get_untrusted_types(file="Modelo/pipeline.skops")

MODELO_PATH = "Modelo/pipeline.skops"
pipeline = load(MODELO_PATH, trusted=unsafe)

UMBRAL = 0.3

FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

# Rangos típicos para los sliders (min, max, default, step)
RANGOS = {
    "radius_mean": (6.0, 30.0, 14.0, 0.1),
    "texture_mean": (9.0, 40.0, 19.0, 0.1),
    "perimeter_mean": (40.0, 190.0, 92.0, 0.5),
    "area_mean": (140.0, 2500.0, 654.0, 5.0),
    "smoothness_mean": (0.05, 0.17, 0.096, 0.001),
    "compactness_mean": (0.02, 0.35, 0.104, 0.001),
    "concavity_mean": (0.0, 0.43, 0.089, 0.001),
    "concave points_mean": (0.0, 0.20, 0.048, 0.001),
    "symmetry_mean": (0.10, 0.30, 0.181, 0.001),
    "fractal_dimension_mean": (0.05, 0.10, 0.063, 0.001),
    "radius_se": (0.1, 3.0, 0.4, 0.01),
    "texture_se": (0.3, 5.0, 1.2, 0.01),
    "perimeter_se": (0.7, 22.0, 2.9, 0.1),
    "area_se": (6.0, 550.0, 40.0, 1.0),
    "smoothness_se": (0.001, 0.031, 0.007, 0.001),
    "compactness_se": (0.002, 0.135, 0.025, 0.001),
    "concavity_se": (0.0, 0.40, 0.032, 0.001),
    "concave points_se": (0.0, 0.053, 0.012, 0.001),
    "symmetry_se": (0.007, 0.080, 0.021, 0.001),
    "fractal_dimension_se": (0.001, 0.030, 0.004, 0.001),
    "radius_worst": (7.0, 37.0, 16.0, 0.1),
    "texture_worst": (12.0, 50.0, 25.0, 0.1),
    "perimeter_worst": (50.0, 252.0, 107.0, 0.5),
    "area_worst": (180.0, 4250.0, 880.0, 5.0),
    "smoothness_worst": (0.07, 0.22, 0.132, 0.001),
    "compactness_worst": (0.02, 1.06, 0.254, 0.001),
    "concavity_worst": (0.0, 1.25, 0.272, 0.001),
    "concave points_worst": (0.0, 0.29, 0.115, 0.001),
    "symmetry_worst": (0.15, 0.66, 0.290, 0.001),
    "fractal_dimension_worst": (0.055, 0.208, 0.084, 0.001),
}

GRUPOS = {
    "📐 Medidas Medias": [f for f in FEATURES if f.endswith("_mean")],
    "📏 Error Estándar": [f for f in FEATURES if f.endswith("_se")],
    "⚠️ Valores Peores": [f for f in FEATURES if f.endswith("_worst")],
}

NOMBRES_LEGIBLES = {
    "radius_mean": "Radio medio",
    "texture_mean": "Textura media",
    "perimeter_mean": "Perímetro medio",
    "area_mean": "Área media",
    "smoothness_mean": "Suavidad media",
    "compactness_mean": "Compacidad media",
    "concavity_mean": "Concavidad media",
    "concave points_mean": "Puntos cóncavos medios",
    "symmetry_mean": "Simetría media",
    "fractal_dimension_mean": "Dim. fractal media",
    "radius_se": "Radio SE",
    "texture_se": "Textura SE",
    "perimeter_se": "Perímetro SE",
    "area_se": "Área SE",
    "smoothness_se": "Suavidad SE",
    "compactness_se": "Compacidad SE",
    "concavity_se": "Concavidad SE",
    "concave points_se": "Puntos cóncavos SE",
    "symmetry_se": "Simetría SE",
    "fractal_dimension_se": "Dim. fractal SE",
    "radius_worst": "Radio peor",
    "texture_worst": "Textura peor",
    "perimeter_worst": "Perímetro peor",
    "area_worst": "Área peor",
    "smoothness_worst": "Suavidad peor",
    "compactness_worst": "Compacidad peor",
    "concavity_worst": "Concavidad peor",
    "concave points_worst": "Puntos cóncavos peores",
    "symmetry_worst": "Simetría peor",
    "fractal_dimension_worst": "Dim. fractal peor",
}

# ── SHAP explainer (inicializado una sola vez) ────────────────────────────────
_explainer = None


def get_explainer():
    global _explainer
    if _explainer is None:
        modelo = pipeline.named_steps["modelo"]
        scaler = pipeline.named_steps["scaler"]
        # Datos de referencia sintéticos basados en medias típicas
        ref = np.array([[RANGOS[f][2] for f in FEATURES]])
        ref_scaled = scaler.transform(ref)
        background = np.tile(ref_scaled, (20, 1)) + np.random.normal(
            0, 0.05, (20, len(FEATURES))
        )
        _explainer = shap.KernelExplainer(modelo.predict_proba, background)
    return _explainer


def generar_shap_plot(X_scaled, shap_vals):
    """Genera gráfica de barras horizontal con valores SHAP para la clase Maligno."""
    vals = shap_vals[0]  # (30,)
    nombres = [NOMBRES_LEGIBLES[f] for f in FEATURES]

    # Top 10 por valor absoluto
    idx = np.argsort(np.abs(vals))[-10:]
    top_vals = vals[idx]
    top_names = [nombres[i] for i in idx]

    colors = ["#e05c5c" if v > 0 else "#4a90d9" for v in top_vals]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars = ax.barh(top_names, top_vals, color=colors, edgecolor="none", height=0.6)

    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel(
        "Valor SHAP (impacto en riesgo de malignidad)", color="#aaa", fontsize=9
    )
    ax.tick_params(colors="#ccc", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Top 10 factores más influyentes", color="white", fontsize=11, pad=10)

    # Leyenda
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e05c5c", label="↑ Aumenta riesgo"),
        Patch(facecolor="#4a90d9", label="↓ Reduce riesgo"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        facecolor="#1a1d27",
        edgecolor="#333",
        labelcolor="#ccc",
        fontsize=8,
    )

    plt.tight_layout()
    return fig


def predecir(*valores):
    X = pd.DataFrame([dict(zip(FEATURES, valores))])
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X)

    proba = pipeline.predict_proba(X)[:, 1][0]
    prediccion = int(proba >= UMBRAL)

    # Resultado
    if prediccion == 1:
        etiqueta = "🔴 MALIGNO"
        color_html = "#e05c5c"
        descripcion = "El modelo detecta características asociadas a tumor maligno."
    else:
        etiqueta = "🟢 BENIGNO"
        color_html = "#4caf7d"
        descripcion = (
            "El modelo no detecta características de malignidad significativas."
        )

    resultado_html = f"""
    <div style="background:#1a1d27;border-radius:12px;padding:20px;text-align:center;border:1px solid #2a2d3a;">
        <div style="font-size:2rem;font-weight:700;color:{color_html};margin-bottom:8px;">{etiqueta}</div>
        <div style="font-size:1.4rem;color:#eee;">Probabilidad de malignidad: <b style="color:{color_html};">{proba:.1%}</b></div>
        <div style="margin-top:8px;color:#aaa;font-size:0.9rem;">Umbral de decisión: {UMBRAL} · {descripcion}</div>
    </div>
    """

    # SHAP
    try:
        explainer = get_explainer()
        shap_vals = explainer.shap_values(X_scaled)
        # shap_vals puede ser (1,30,2) o lista
        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[1])  # clase Maligno
        else:
            sv = shap_vals[:, :, 1]
        fig = generar_shap_plot(X_scaled, sv)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        ax.text(
            0.5,
            0.5,
            f"SHAP no disponible:\n{str(e)}",
            ha="center",
            va="center",
            color="white",
            transform=ax.transAxes,
        )
        ax.axis("off")

    return resultado_html, fig


# ── UI ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

body, .gradio-container {
    background: #0b0d14 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #e0e0e0 !important;
}

.gr-panel, .gr-box, .gr-form {
    background: #13151f !important;
    border: 1px solid #1f2235 !important;
    border-radius: 10px !important;
}

h1, h2, h3, .gr-block-label {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #e0e0e0 !important;
}

.gr-button-primary {
    background: #3d6fff !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}

.gr-slider input[type=range]::-webkit-slider-thumb {
    background: #3d6fff !important;
}

footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Clasificador de Tumor de Mama") as demo:

    gr.HTML(
        """
    <div style="text-align:center;padding:32px 0 16px;border-bottom:1px solid #1f2235;margin-bottom:24px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;letter-spacing:0.2em;color:#3d6fff;margin-bottom:8px;">HERRAMIENTA CLÍNICA DE APOYO</div>
        <h1 style="font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:600;color:#fff;margin:0;">
            Clasificador de Tumor de Mama
        </h1>
        <p style="color:#888;margin-top:10px;font-size:0.9rem;max-width:600px;margin-inline:auto;">
            Introduce las métricas de la biopsia para obtener una predicción y análisis de factores de riesgo.<br>
            <span style="color:#e0a020;">⚠ Solo para uso investigativo. No reemplaza criterio médico.</span>
        </p>
    </div>
    """
    )

    with gr.Row():
        # ── Panel izquierdo: inputs ──────────────────────────────────────────
        with gr.Column(scale=2):
            inputs = []
            for grupo, features in GRUPOS.items():
                with gr.Accordion(grupo, open=(grupo == "📐 Medidas Medias")):
                    with gr.Row():
                        for i, feat in enumerate(features):
                            mn, mx, default, step = RANGOS[feat]
                            sl = gr.Slider(
                                minimum=mn,
                                maximum=mx,
                                value=default,
                                step=step,
                                label=NOMBRES_LEGIBLES[feat],
                            )
                            inputs.append(sl)

            btn = gr.Button("🔬 Analizar muestra", variant="primary", size="lg")

        # ── Panel derecho: outputs ───────────────────────────────────────────
        with gr.Column(scale=3):
            resultado = gr.HTML(
                value="""<div style="background:#1a1d27;border-radius:12px;padding:30px;text-align:center;border:1px solid #2a2d3a;color:#555;">
                    Configura los parámetros y presiona <b>Analizar muestra</b>
                </div>"""
            )
            gr.HTML(
                "<div style='margin-top:16px;font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#555;letter-spacing:0.1em;'>EXPLICABILIDAD · SHAP</div>"
            )
            shap_plot = gr.Plot()

    btn.click(fn=predecir, inputs=inputs, outputs=[resultado, shap_plot])

    gr.HTML(
        """
    <div style="text-align:center;padding:20px 0;margin-top:24px;border-top:1px solid #1f2235;">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#333;letter-spacing:0.15em;">
            MODELO: MLPClassifier · DATASET: Breast Cancer Wisconsin · UMBRAL: 0.3
        </span>
    </div>
    """
    )

if __name__ == "__main__":
    demo.launch()
