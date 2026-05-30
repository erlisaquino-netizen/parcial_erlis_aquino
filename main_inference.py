"""
Inferencia producción - parcial
"""

import joblib

from src.preprocessing import preprocess

from src.postprocessing import (
    postprocess,
    export_results
)
 
import pandas as pd


MODEL_PATH = "models/best_model.pkl"

DATA_PATH = "data/raw/p10_extrac.csv"


def main():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    df = preprocess(df)

    #X = df.drop(columns=["target"])
   
    X = df.drop(columns=["target", "partition"])

    probs = model.predict_proba(X)[:, 1]

    results = postprocess(
        df,
        probs
    )

    export_results(results)

    print("Inferencia completada")

    print(results.head(10))
if __name__ == "__main__":

    main()