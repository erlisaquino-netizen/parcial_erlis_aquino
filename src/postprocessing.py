"""
Posprocesamiento
"""

import numpy as np
import pandas as pd


FRESCURA_MAP = {
    "G1": 0.066,
    "G2": 0.028,
    "G3": 0.022,
    "G4": 0.008
}  
 
FRESCURA_DEFAULT = 0.004


def assign_frescura(df):

    return df["grp_campecs06m"].map(
        FRESCURA_MAP
    ).fillna(FRESCURA_DEFAULT)


def compute_score(df, probs):

    frescura = assign_frescura(df)

    score = (
        probs
        * df["prob_value_contact"]
        * np.log(df["monto"] + 1)
        * frescura
    )

    return score


def segment_groups(score):

    return pd.qcut(
        score,
        q=5,
        labels=[5, 4, 3, 2, 1]
    )


def postprocess(df, probs):

    result = df.copy()

    result["prob_model"] = probs

    result["score"] = compute_score(
        result,
        probs
    )

    result["priority_group"] = segment_groups(
        result["score"]
    )

    return result.sort_values(
        "score",
        ascending=False
    )


def export_results(df):

    output_path = "data/predictions/predictions.csv"

    df.to_csv(output_path, index=False)

    print(f"Exportado: {output_path}")