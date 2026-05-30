"""
TAREA 0: Pipeline de Machine Learning Básico
==============================================
Rúbrica de evaluación:
- Código completo (todas las etapas funcionando) : 10 puntos
- Código ordenado (estructura modular clara)     : 7 puntos
- Buenas prácticas (funciones, docstrings, PEP8) : 3 puntos


Dataset: cargado desde DIRECCIÓN DRIVE - parcial
"""
import src.preprocessing as pp
import src.training as tr
import src.postprocessing as post

   
DATA_PATH = "DIRECCIÓN DRIVE"  # <-- reemplazar con ruta real


def main():
    # 1. Cargar y preprocesar
    df_raw = pp.load_data(DATA_PATH)
    df_processed = pp.preprocess(df_raw)


    # 2. Entrenamiento dummy
    model, predictions = tr.train_dummy(df_processed)


    # 3. Posprocesamiento
    results = post.postprocess(df_processed, predictions)
    post.export_results(results)
 