"""Stage 3: attention DNN + Isolation Forest; write hybrid_dnn_anomaly_preds.csv."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from fraud_ml.config.paths import get_paths

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
TARGET_COL = "isFraud"


def run(project_root: Path | None = None, save_plots: bool = True) -> None:
    paths = get_paths(project_root)
    project_root = paths.root
    processed_dir = paths.processed_dir
    figures_dir = paths.reports_figures

    ieee_candidates = [
        processed_dir / "ieee_train_eda_ready.csv",
        processed_dir / "ieee_train_merged_cleaned.csv",
        project_root / "ieee_train_eda_ready.csv",
    ]
    ieee_path = next((p for p in ieee_candidates if p.exists()), None)
    if ieee_path is None:
        raise FileNotFoundError("Could not find cleaned IEEE training file (ieee_train_eda_ready.csv).")

    df = pd.read_csv(ieee_path)
    print(f"Loaded IEEE data: {ieee_path}")
    print("IEEE shape:", df.shape)

    elliptic_clean_path = processed_dir / "elliptic_transactions_cleaned.csv"
    if elliptic_clean_path.exists():
        elliptic_df = pd.read_csv(elliptic_clean_path)
        print(f"Loaded Elliptic reference data: {elliptic_clean_path} -> {elliptic_df.shape}")
    else:
        print("Elliptic cleaned file not found in processed_data (optional for this stage).")

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in IEEE dataset.")

    gbdt_candidates = [processed_dir / "gbdt_preds.csv", project_root / "gbdt_preds.csv"]
    gbdt_path = next((p for p in gbdt_candidates if p.exists()), None)
    if gbdt_path is None:
        raise FileNotFoundError("Could not find gbdt_preds.csv from stage 2.")

    gbdt_preds = pd.read_csv(gbdt_path)
    print(f"Loaded GBDT predictions: {gbdt_path} -> {gbdt_preds.shape}")
    if "gbdt_pred_proba" not in gbdt_preds.columns:
        raise KeyError("gbdt_preds.csv must contain a 'gbdt_pred_proba' column.")

    df_model = df.copy()
    if "TransactionID" in df_model.columns and "TransactionID" in gbdt_preds.columns:
        df_model = df_model.merge(
            gbdt_preds[["TransactionID", "gbdt_pred_proba"]], on="TransactionID", how="left"
        )
        print("Merged gbdt_pred_proba by TransactionID.")
    else:
        df_model["gbdt_pred_proba"] = np.nan
        n = min(len(df_model), len(gbdt_preds))
        df_model.loc[df_model.index[:n], "gbdt_pred_proba"] = gbdt_preds["gbdt_pred_proba"].values[:n]
        print("Merged gbdt_pred_proba by row order fallback.")

    df_model["gbdt_pred_available"] = (~df_model["gbdt_pred_proba"].isna()).astype(int)
    df_model["gbdt_pred_proba"] = df_model["gbdt_pred_proba"].fillna(df_model["gbdt_pred_proba"].median())
    print("GBDT feature null count after merge:", df_model["gbdt_pred_proba"].isna().sum())
    print("GBDT merge coverage:", df_model["gbdt_pred_available"].mean())

    id_col = "TransactionID" if "TransactionID" in df_model.columns else None
    feature_cols = [c for c in df_model.columns if c != TARGET_COL]
    X_num = df_model[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df_model[TARGET_COL].astype(int).values
    ids_all = df_model[id_col].copy() if id_col is not None else pd.Series(np.arange(len(df_model)), name="row_id")

    print("Numeric feature matrix:", X_num.shape)
    print("Fraud ratio:", y.mean())

    X_train, X_valid, y_train, y_valid, ids_train, ids_valid = train_test_split(
        X_num, y, ids_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_all_imp = imputer.transform(X_num)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_valid_scaled = scaler.transform(X_valid_imp)
    X_all_scaled = scaler.transform(X_all_imp)
    print("Train/valid shapes:", X_train_scaled.shape, X_valid_scaled.shape)

    dnn_valid_proba: np.ndarray
    dnn_all_proba: np.ndarray

    try:
        import tensorflow as tf
        from tensorflow.keras import Model, callbacks, layers

        tf.random.set_seed(RANDOM_STATE)
        n_features = X_train_scaled.shape[1]

        def build_attention_dnn(n_features: int, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.2):
            inp = layers.Input(shape=(n_features,), name="features")
            x = layers.Reshape((n_features, 1))(inp)
            x = layers.Dense(embed_dim, activation="relu")(x)
            attn_out = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=max(8, embed_dim // num_heads),
                dropout=dropout,
                name="feature_attention",
            )(x, x)
            x = layers.Add()([x, attn_out])
            x = layers.LayerNormalization()(x)
            x = layers.Flatten()(x)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Dense(64, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
            out = layers.Dense(1, activation="sigmoid", name="fraud_prob")(x)
            model = Model(inputs=inp, outputs=out)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
                metrics=[tf.keras.metrics.AUC(name="auc")],
            )
            return model

        dnn_model = build_attention_dnn(n_features=n_features)
        dnn_model.summary()

        early_stop = callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True, verbose=1
        )
        history = dnn_model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_valid_scaled, y_valid),
            epochs=40,
            batch_size=1024,
            callbacks=[early_stop],
            verbose=1,
        )

        plt.figure(figsize=(10, 4))
        plt.plot(history.history["auc"], label="train_auc")
        plt.plot(history.history["val_auc"], label="val_auc")
        plt.title("Attention-DNN AUC by epoch")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()
        plt.tight_layout()
        if save_plots:
            figures_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(figures_dir / "stage03_dnn_auc.png", dpi=120, bbox_inches="tight")
        plt.close()

        dnn_valid_proba = dnn_model.predict(X_valid_scaled, verbose=0).ravel()
        dnn_all_proba = dnn_model.predict(X_all_scaled, verbose=0).ravel()
        dnn_label = "Attention-DNN (TensorFlow)"
    except (ImportError, OSError) as e:
        print(
            "TensorFlow not installed or incompatible with this Python version; "
            "using sklearn MLPClassifier fallback for dnn_pred_proba.\n"
            "For the attention DNN, use Python 3.10–3.12 and: pip install 'tensorflow>=2.15'\n"
            f"  ({e})"
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=1024,
            learning_rate_init=1e-3,
            max_iter=40,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=RANDOM_STATE,
        )
        mlp.fit(X_train_scaled, y_train)
        if hasattr(mlp, "loss_curve_") and mlp.loss_curve_ is not None and len(mlp.loss_curve_) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(mlp.loss_curve_, label="train_loss")
            plt.title("MLPClassifier loss (sklearn fallback)")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            if save_plots:
                figures_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(figures_dir / "stage03_dnn_auc.png", dpi=120, bbox_inches="tight")
            plt.close()
        dnn_valid_proba = mlp.predict_proba(X_valid_scaled)[:, 1]
        dnn_all_proba = mlp.predict_proba(X_all_scaled)[:, 1]
        dnn_label = "MLPClassifier (sklearn fallback)"

    dnn_auc = roc_auc_score(y_valid, dnn_valid_proba)
    print(f"{dnn_label} — Validation ROC-AUC: {dnn_auc:.4f}")

    dnn_valid_pred = (dnn_valid_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_valid, dnn_valid_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Fraud (1)"])
    disp.plot(cmap="Blues")
    plt.title("DNN / MLP — Confusion Matrix (Validation)")
    if save_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / "stage03_dnn_confusion.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(classification_report(y_valid, dnn_valid_pred, digits=4))

    fraud_rate = float(np.mean(y_train))
    iso = IsolationForest(
        n_estimators=300,
        contamination=max(0.01, min(0.20, fraud_rate)),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train_scaled)
    anomaly_score_all = -iso.score_samples(X_all_scaled)
    anomaly_score_valid = -iso.score_samples(X_valid_scaled)
    anom_scaler = MinMaxScaler()
    anomaly_score_all_norm = anom_scaler.fit_transform(anomaly_score_all.reshape(-1, 1)).ravel()
    anomaly_score_valid_norm = anom_scaler.transform(anomaly_score_valid.reshape(-1, 1)).ravel()
    anomaly_auc_valid = roc_auc_score(y_valid, anomaly_score_valid_norm)
    print(f"Isolation Forest anomaly-score ROC-AUC (validation): {anomaly_auc_valid:.4f}")

    valid_plot_df = pd.DataFrame({"isFraud": y_valid, "anomaly_score": anomaly_score_valid_norm})
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=valid_plot_df, x="anomaly_score", hue="isFraud", fill=True, common_norm=False, alpha=0.35)
    plt.title("Anomaly Score Density (Validation): Fraud vs Normal")
    plt.xlabel("Normalized anomaly score (higher = more anomalous)")
    if save_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / "stage03_anomaly_kde.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(valid_plot_df.groupby("isFraud")["anomaly_score"].describe())

    w_dnn, w_anom = 0.7, 0.3
    hybrid_score_all = w_dnn * dnn_all_proba + w_anom * anomaly_score_all_norm
    hybrid_df = pd.DataFrame(
        {
            "dnn_pred_proba": dnn_all_proba,
            "anomaly_score": anomaly_score_all_norm,
            "hybrid_score": hybrid_score_all,
            "isFraud": y,
        }
    )
    if id_col is not None:
        hybrid_df.insert(0, "TransactionID", ids_all.values)

    hybrid_out_path = processed_dir / "hybrid_dnn_anomaly_preds.csv"
    processed_dir.mkdir(parents=True, exist_ok=True)
    hybrid_df.to_csv(hybrid_out_path, index=False)
    print(f"Saved hybrid predictions: {hybrid_out_path}")
    print(hybrid_df.head())


def main() -> None:
    run()


if __name__ == "__main__":
    main()
