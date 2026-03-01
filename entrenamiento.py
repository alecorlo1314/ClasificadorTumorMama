from src.datos import cargar_datos, separar_features, dividir_datos
from src.entrenar import comparar_modelos
from src.guardar import guardar_modelo
from src.evaluar import evaluar_modelo, generar_reporte
from src.explicar import explicar_modelo

df = cargar_datos("Datos/breast_cancer_diagnostic_dataset.csv")
X, y = separar_features(df)
X_train, X_test, y_train, y_test = dividir_datos(X, y)

pipeline, mejor_modelo = comparar_modelos(X_train, y_train)
pipeline.fit(X_train, y_train)

guardar_modelo(pipeline, "Modelo/pipeline.skops")
f1, precision, recall, accuracy = evaluar_modelo(pipeline, X_test, y_test, umbral=0.3)
explicar_modelo(pipeline, X_test)
generar_reporte(f1, precision, recall, accuracy)

print(f"Entrenamiento completado con {mejor_modelo}")
