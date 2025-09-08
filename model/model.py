# model/model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def detect_anomalies(df: pd.DataFrame, method: str = "isolation_forest"):
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns.")

    X = df.select_dtypes(include=[np.number]).dropna()
    if X.empty:
        raise ValueError("No numeric data found in CSV.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "isolation_forest":
        model = IsolationForest(contamination=0.1, random_state=42)
    elif method == "svm":
        model = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    elif method == "zscore":
        # Simple Z-score method
        zscores = np.abs((X - X.mean()) / X.std())
        anomalies = (zscores > 3).any(axis=1)
        df["anomaly"] = anomalies.astype(int)
        return df, plot_data(X, anomalies)
    else:
        raise ValueError("Unsupported method")

    preds = model.fit_predict(X_scaled)
    df["anomaly"] = (preds == -1).astype(int)
    return df, plot_data(X, preds == -1)


def plot_data(X, anomalies):
    fig, ax = plt.subplots()
    if X.shape[1] >= 2:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=~anomalies, cmap="coolwarm", label="Normal")
        ax.scatter(X.iloc[:, 0][anomalies], X.iloc[:, 1][anomalies], c='red', label="Anomaly", marker='x')
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
    else:
        ax.plot(X.iloc[:, 0], label="Value")
        ax.scatter(X.index[anomalies], X.iloc[:, 0][anomalies], c="red", label="Anomaly", marker='x')

    ax.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
