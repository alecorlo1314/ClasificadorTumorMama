from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


def construir_pipeline(modelo):
    return Pipeline(
        [
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("modelo", modelo),
        ]
    )


def comparar_modelos(X_train, y_train):
    modelos = {
        "MLPClassifier": MLPClassifier(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        pipeline = construir_pipeline(modelo)
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
        resultados[nombre] = round(np.mean(scores), 4)
        print(f"{nombre}: F1 promedio = {resultados[nombre]}")

    mejor = max(resultados, key=resultados.get)
    print(f"\nMejor modelo: {mejor} con F1 = {resultados[mejor]}")
    return construir_pipeline(modelos[mejor]), mejor
