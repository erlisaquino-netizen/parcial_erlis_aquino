"""
Entrenamiento con Optuna + LightGBM -- parcial
"""

import optuna
import lightgbm as lgb
import mlflow
from sklearn.metrics import roc_auc_score
import mlflow.lightgbm
 

TARGET_COL = "target"


def split_temporal(df):

    train_df = df[df["partition"].isin(["p1","p2","p3","p4","p5","p6","p7"])]
    valid_df = df[df["partition"] == "p8"]
    test_df  = df[df["partition"] == "p9"]

    return train_df, valid_df, test_df


def objective(trial, X_train, y_train, X_valid, y_valid):

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 20, 120),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300)
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_valid)[:, 1]

    return roc_auc_score(y_valid, preds)

 
def train_model(df):

    mlflow.set_experiment("pipeline_mlo_ps_lightgbm")

    train_df, valid_df, test_df = split_temporal(df)

    X_train = train_df.drop(columns=[TARGET_COL, "partition"])
    y_train = train_df[TARGET_COL]

    X_valid = valid_df.drop(columns=[TARGET_COL, "partition"])
    y_valid = valid_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL, "partition"])
    y_test = test_df[TARGET_COL]

    # =========================
    # OPTUNA
    # =========================
    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_valid, y_valid),
        n_trials=20
    )

    best_params = study.best_params
    best_score = study.best_value

    # =========================
    # MODELO FINAL
    # =========================
    model = lgb.LGBMClassifier(
        objective="binary",
        **best_params
    )

    model.fit(X_train, y_train)

    # =========================
    # MÉTRICAS
    # =========================
    train_preds = model.predict_proba(X_train)[:, 1]
    valid_preds = model.predict_proba(X_valid)[:, 1]

    train_auc = roc_auc_score(y_train, train_preds)
    valid_auc = roc_auc_score(y_valid, valid_preds)

    decay = round((train_auc - valid_auc) * 100, 2)

    # =========================
    # MLFLOW
    # =========================
    """with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("valid_auc", valid_auc)
        mlflow.log_metric("decay", decay)
    """


    with mlflow.start_run():

        mlflow.log_params(best_params)
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("valid_auc", valid_auc)
        mlflow.log_metric("decay", decay)

        # 🔥 MODELO VERSIONADO (IMPORTANTE)
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="lightgbm_mlo_ps_model"
        )


    # =========================
    # GUARDAR MODELO
    # =========================
    import joblib
    joblib.dump(model, "models/best_model.pkl")

    print(f"AUC VALIDATION: {valid_auc:.4f}")
  
    return (
        model,
        valid_preds,
        valid_df,
        valid_auc,
        best_params,
        best_score,
        train_auc,
        valid_auc,
        decay
    )  