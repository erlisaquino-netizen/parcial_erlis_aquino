"""
Preprocesamiento - parcial
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder


DROP_COLUMNS = [
    "key_value",
    "fch_creacion"
]
 

def load_multiple_files(paths):

    dfs = []

    for path in paths:

        df = pd.read_csv(path)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def handle_missing(df):

    numeric_cols = df.select_dtypes(
        include=["number"]
    ).columns

    for col in numeric_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

    return df


def encode_categoricals(df):

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_cols:

        if col != "partition":

            le = LabelEncoder()

            df[col] = le.fit_transform(
                df[col].astype(str)
            )

    return df
 

def preprocess(df):

    df = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    df = handle_missing(df)

    df = encode_categoricals(df)

    return df