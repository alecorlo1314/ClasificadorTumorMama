import pandas as pd
from sklearn.model_selection import train_test_split


def cargar_datos(ruta: str):
    df = pd.read_csv(ruta)
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df


def separar_features(df: pd.DataFrame):
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y


def dividir_datos(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
