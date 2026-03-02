# 🔬 Clasificador de Tumor de Mama

Proyecto de Machine Learning para clasificar tumores de mama como **benignos o malignos** usando el dataset Breast Cancer Wisconsin. Incluye pipeline completo con CI/CD, versionado de datos con DVC y explicabilidad con SHAP.

🚀 **[Demo en vivo en Hugging Face Spaces](https://huggingface.co/spaces/alecorlo1234/ClasificadorTumorMama)**

---

## 📊 Resultados del Modelo

| Métrica | Valor |
|---|---|
| Algoritmo | MLPClassifier |
| F1-Score (CV) | 0.9708 |
| Umbral de decisión | 0.3 |
| Dataset | 569 muestras · 30 features |

> El umbral se ajustó a 0.3 (en lugar del 0.5 estándar) para **maximizar el recall** y reducir falsos negativos — en contexto clínico es preferible sobrediagnosticar que pasar por alto un tumor maligno.

---

## 🏗️ Arquitectura del Proyecto

```
ClasificadorTumorMama/
├── src/
│   ├── datos.py        # Carga y división del dataset
│   ├── entrenar.py     # Comparación de algoritmos + selección del mejor
│   ├── evaluar.py      # Métricas, matriz de confusión, curva ROC
│   ├── explicar.py     # SHAP summary plot
│   └── guardar.py      # Serialización del modelo con skops
├── Aplicacion/
│   ├── tumor_app.py    # App Gradio con inputs clínicos y SHAP local
│   ├── requirements.txt
│   └── README.md       # Configuración para Hugging Face Spaces
├── Modelo/
│   └── pipeline.skops  # Pipeline serializado (SMOTE + Scaler + MLP)
├── Datos/              # Gestionado por DVC (no en Git)
├── Resultados/         # Generado en CI (no en Git)
├── .github/
│   ├── workflows/ci.yml   # Entrenamiento + reporte automático
│   └── workflows/cd.yml   # Deploy a Hugging Face
├── entrenamiento.py    # Script principal del pipeline
├── Makefile
└── requirements.txt
```

---

## ⚙️ Pipeline de CI/CD

```
Push a main
    │
    ▼
┌─────────────────────────────────────┐
│         Continuous Integration       │
│  format → lint → DVC pull → train   │
│  → eval → reporte en PR → push      │
│         al branch update            │
└──────────────────┬──────────────────┘
                   │ éxito
                   ▼
┌─────────────────────────────────────┐
│        Continuous Deployment         │
│  checkout update → login HF →       │
│  upload Aplicacion/ + Modelo/        │
│       a Hugging Face Spaces          │
└─────────────────────────────────────┘
```

---

## 🚀 Instalación y uso local

### 1. Clonar el repositorio

```bash
git clone https://github.com/alecorlo1314/ClasificadorTumorMama.git
cd ClasificadorTumorMama
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar DVC y descargar datos

Necesitas una cuenta en [DagsHub](https://dagshub.com).

```bash
export DAGSHUB_TOKEN=tu_token_aqui

make configuracion_DVC_remoto
dvc remote modify tumor_storage password $DAGSHUB_TOKEN
dvc pull -r tumor_storage
```

### 4. Entrenar el modelo

```bash
make train
```

Esto compara 4 algoritmos (MLP, RandomForest, LogisticRegression, XGBoost) via cross-validation y guarda el mejor en `Modelo/pipeline.skops`.

### 5. Evaluar

```bash
make eval
```

Genera en `Resultados/`: métricas en texto, matriz de confusión, curva ROC y SHAP summary.

### 6. Correr la app localmente

```bash
cd Aplicacion
pip install -r requirements.txt
python tumor_app.py
```

---

## 🔐 Secrets de GitHub necesarios

Para que el pipeline CI/CD funcione en tu fork, configura estos secrets en **Settings → Secrets and variables → Actions**:

| Secret | Descripción |
|---|---|
| `DAGSHUB_TOKEN` | Token de API de DagsHub |
| `HF_TUMOR` | Token de Hugging Face con permisos de escritura |
| `USER_NAME` | Tu nombre para los commits automáticos |
| `USER_EMAIL` | Tu email para los commits automáticos |

> `GITHUB_TOKEN` se genera automáticamente, no es necesario crearlo.

---

## 📋 Comandos disponibles (Makefile)

```bash
make install              # Instalar dependencias
make format               # Verificar formato con black
make lint                 # Analizar calidad con pylint
make train                # Entrenar modelo
make eval                 # Evaluar y generar reporte
make configuracion_DVC_remoto  # Configurar remote de DagsHub
make deploy HF=<token>    # Deploy manual a Hugging Face
```

---

## 🧪 Algoritmos comparados

| Algoritmo | F1-Score (CV) |
|---|---|
| **MLPClassifier** ✅ | **0.9708** |
| LogisticRegression | 0.9583 |
| XGBoost | 0.9480 |
| RandomForest | 0.9467 |

La selección es automática — si en el futuro un algoritmo diferente supera al MLP, el pipeline lo elegirá sin cambios manuales.

---

## 🔍 Explicabilidad con SHAP

El proyecto incluye dos niveles de explicabilidad:

- **Global** (`src/explicar.py`): SHAP summary plot con las features más importantes en el conjunto de test. Se publica automáticamente en el reporte de cada PR.
- **Local** (app Gradio): Para cada predicción individual, muestra los top 10 factores que más influyeron en ese resultado específico, con dirección (aumenta / reduce riesgo de malignidad).

---

## 📦 Tecnologías utilizadas

- **ML**: scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Explicabilidad**: SHAP
- **Versionado de datos**: DVC + DagsHub
- **Serialización**: skops
- **App**: Gradio
- **CI/CD**: GitHub Actions + CML
- **Deploy**: Hugging Face Spaces

---

## ⚠️ Aviso

Este proyecto es de carácter **educativo e investigativo**. Las predicciones del modelo no deben usarse como sustituto de un diagnóstico médico profesional.
