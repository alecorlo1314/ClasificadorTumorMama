import skops.io as sio


def guardar_modelo(pipeline, ruta: str):
    sio.dump(pipeline, ruta)
    print(f"Modelo guardado en {ruta}")


def cargar_modelo(ruta: str):
    unknown = sio.get_untrusted_types(file=ruta)
    return sio.load(ruta, trusted=unknown)
