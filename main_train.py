"""
Pipeline entrenamiento MLOps completo -parcial
"""
"""
evita hordcodeo
"""
import psutil
import time
import json
import glob

from src.preprocessing import load_multiple_files, preprocess
from src.training import train_model
from src.postprocessing import postprocess, export_results


# =========================
# CONFIGURACIÓN DINÁMICA
# =========================
TRAIN_FILES = sorted(
    glob.glob("data/raw/p*_extrac.csv")
)


def main():

    # =========================
    # MONITOREO INICIO
    # =========================
    inicio = time.time()
    cpu_inicio = psutil.cpu_percent()
    ram_inicio = psutil.virtual_memory().percent

    # =========================
    # INGESTA
    # =========================
    print("Cargando datasets...")
    print(f"Archivos encontrados: {len(TRAIN_FILES)}")

    df = load_multiple_files(TRAIN_FILES)

    # =========================
    # PREPROCESAMIENTO
    # =========================
    print("Preprocesando...")
    df = preprocess(df)

    # =========================
    # ENTRENAMIENTO
    # =========================
    print("Entrenando modelo...")

    (
        model,
        preds,
        valid_df,
        auc,
        best_params,
        best_score,
        train_auc,
        valid_auc,
        decay
    ) = train_model(df)

    # =========================
    # POSTPROCESAMIENTO
    # =========================
    print("Posprocesando...")
    results = postprocess(valid_df, preds)
    export_results(results)

    # =========================
    # MONITOREO FINAL
    # =========================
    cpu_fin = psutil.cpu_percent()
    ram_fin = psutil.virtual_memory().percent
    fin = time.time()

    tiempo_total = round(fin - inicio, 2)

    # =========================
    # MÉTRICAS
    # =========================
    metrics = {
        "auc": auc,
        "train_auc": train_auc,
        "valid_auc": valid_auc,
        "decay": decay,
        "best_params": best_params,
        "optuna_best_score": best_score,
        "model_name": "LightGBM",
        "model_path": "models/best_model.pkl",
        "cpu_inicio": cpu_inicio,
        "cpu_final": cpu_fin,
        "ram_inicio": ram_inicio,
        "ram_final": ram_fin,
        "tiempo_total": tiempo_total
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n===== MONITOREO =====")
    print(metrics)

    print("Métricas exportadas: metrics.json")
    print("Pipeline completado")


if __name__ == "__main__":
    main()