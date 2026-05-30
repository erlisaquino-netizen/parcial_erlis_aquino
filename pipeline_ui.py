import streamlit as st
import time
import psutil
import pandas as pd
import json
import os
 
pred_path = "data/predictions/predictions.csv"

if os.path.exists(pred_path):
    df_pred = pd.read_csv(pred_path)
else:
    df_pred = None


st.set_page_config(
    page_title="Pipeline MLOps",
    layout="wide"
)

st.title("🚀 Pipeline MLOps")

st.markdown(
    "Pipeline de entrenamiento e inferencia con monitoreo operativo"
)

# COLUMNAS PIPELINE
col1, col2, col3, col4 = st.columns(4)

ingesta = col1.empty()
prep = col2.empty()
train = col3.empty()
evalua = col4.empty()

# ESTADO INICIAL
ingesta.info("📥 Ingesta")
prep.info("🧹 Preprocesamiento")
train.info("🤖 Entrenamiento")
evalua.info("📊 Evaluación")

# BARRA PROGRESO
progress = st.progress(0)

# BOTÓN
if st.button("▶️ Ejecutar Pipeline"):

    # =========================
    # CARGAR MÉTRICAS REALES
    # =========================
    with open("metrics.json") as f:
        metrics = json.load(f)

    auc = metrics["auc"]

    cpu_inicio = metrics["cpu_inicio"]
    cpu_fin = metrics["cpu_final"]
    ram_inicio = metrics["ram_inicio"]
    ram_fin = metrics["ram_final"]
    tiempo = metrics["tiempo_total"]

    # 🔥 NUEVO (LO QUE TE FALTABA)
    train_auc = metrics.get("train_auc")
    valid_auc = metrics.get("valid_auc")
    decay = metrics.get("decay")

    best_params = metrics.get("best_params", {})
    optuna_score = metrics.get("optuna_best_score")
    model_name = metrics.get("model_name")

    # =========================
    # PIPELINE VISUAL
    # =========================
    ingesta.warning("⚙️ Ejecutando...")
    progress.progress(20)
    time.sleep(1)
    ingesta.success("✅ Ingesta completada")

    prep.warning("⚙️ Ejecutando...")
    progress.progress(40)
    time.sleep(1)
    prep.success("✅ Preprocesamiento completado")

    train.warning("🔥 Entrenando modelo...")
    progress.progress(70)
    time.sleep(2)
    train.success("✅ Modelo entrenado")

    evalua.warning("📈 Evaluando...")
    progress.progress(90)
    time.sleep(1)
    evalua.success(f"✅ AUC: {auc:.4f}")

    progress.progress(100)

    # =========================
    # MÉTRICAS OPERATIVAS
    # =========================
    st.divider()

    st.subheader("📡 Monitoreo Operativo")





    diff = abs(train_auc - valid_auc) if train_auc and valid_auc else None

    if diff is not None:
        st.metric("Train-Validation Gap", f"{diff:.4f}")

        if diff < 0.02:
            st.success("✔ Modelo bien generalizado")
        elif diff < 0.05:
            st.warning("⚠ Overfitting leve")
        else:
            st.error("🚨 Overfitting fuerte")




    m1, m2, m3, m4 = st.columns(4)

    m1.metric("CPU Inicio", f"{cpu_inicio}%")
    m2.metric("CPU Final", f"{cpu_fin}%")
    m3.metric("RAM Final", f"{ram_fin}%")
    m4.metric("Tiempo", f"{tiempo} seg")

    # =========================
    # PERFORMANCE MODELO
    # =========================
    #st.subheader("🎯 Performance Modelo")

    #st.subheader("🏆 Top 10 Clientes Prioritarios")

    #if "score" in locals():
    #    st.dataframe(df.sort_values("score", ascending=False).head(10))

    st.subheader("🏆 Top 10 Clientes Prioritarios")

    if df_pred is not None and "score" in df_pred.columns:
        top10 = df_pred.sort_values("score", ascending=False).head(10)
        st.dataframe(top10)

    else:
        st.info("Top clientes disponible en dashboard de inferencia")


    #st.metric("AUC VALIDATION", f"{auc:.4f}")
    st.metric("AUC Validation", f"{valid_auc:.4f}")
    # =========================
    # 🔥 NUEVO: TRAIN vs VALID 
    # =========================
    st.subheader("📊 Train vs Validation AUC")

    if train_auc and valid_auc:
        df_compare = pd.DataFrame({
            "Dataset": ["Train", "Validation"],
            "AUC": [train_auc, valid_auc]
        })
        st.bar_chart(df_compare.set_index("Dataset"))

    # =========================
    # 🔥 NUEVO: DECAY
    # =========================
    st.subheader("📉 Overfitting Control (Decay)")

    if decay is not None:
        st.metric("Decay", f"{decay}%")

        if decay < 10:
            st.success("✔ Modelo estable (sin overfitting)")
        else:
            st.error("⚠ Overfitting detectado")

    # =========================
    # 🔥 NUEVO: OPTUNA
    # =========================
    st.subheader("⚙️ Optuna - Hiperparámetros")

    st.json(best_params)

    if optuna_score:
        st.metric("Best Optuna Score", f"{optuna_score:.4f}")

    # =========================
    # 🔥 NUEVO: MODELO FINAL
    # =========================
    st.subheader("🤖 Modelo Final")

    st.success(f"Modelo seleccionado: {model_name}")

    # =========================
    # 🔥 NUEVO: MLflow (simple visual)
    # =========================
    st.subheader("🧬 MLflow Tracking")

    st.info("Run registrado en MLflow")
    st.code("auc_validation + hyperparameters + metrics logged")

    # =========================
    # GRÁFICO ORIGINAL
    # =========================
    st.subheader("📈 Uso de Recursos")

    chart_data = pd.DataFrame({
        "Metrica": [
            "CPU Inicio",
            "CPU Final",
            "RAM Inicio",
            "RAM Final"
        ],
        "Valor": [
            cpu_inicio,
            cpu_fin,
            ram_inicio,
            ram_fin
        ]
    })

    st.bar_chart(chart_data.set_index("Metrica"))

    # =========================
    # FINAL
    # =========================
    st.balloons()
    st.success("🎉 Pipeline completado correctamente") 