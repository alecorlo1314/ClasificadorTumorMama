import shap
import matplotlib.pyplot as plt


def explicar_modelo(pipeline, X_test):
    modelo = pipeline.named_steps["modelo"]
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X_test)

    explainer = shap.KernelExplainer(modelo.predict_proba, shap.sample(X_scaled, 50))
    shap_values = explainer.shap_values(X_scaled[:50])

    # shap_values tiene forma (50, 30, 2) — tomamos la clase Maligno (índice 1)
    shap_maligno = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(
        shap_maligno, X_scaled[:50], feature_names=X_test.columns.tolist(), show=False
    )
    plt.tight_layout()
    plt.savefig("Resultados/shap_summary.png")
    plt.close()
    print("SHAP summary guardado en Resultados/")
