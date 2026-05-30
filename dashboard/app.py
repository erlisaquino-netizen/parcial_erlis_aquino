""" 
parcial
"""

import streamlit as st
import pandas as pd


st.title("Dashboard MLOps")


#df = pd.read_csv(
#    "../data/predictions/predictions.csv"
#)
df = pd.read_csv(
    "data/predictions/predictions.csv"
)

st.subheader("Predicciones")

st.dataframe(df.head())

st.subheader("Distribución Prioridad")

st.bar_chart(
    df["priority_group"].value_counts()
)  