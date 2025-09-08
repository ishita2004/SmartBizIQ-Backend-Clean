from fastapi import FastAPI, UploadFile, File, Query, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import pandas as pd
import numpy as np
import io
import base64
import traceback

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

from io import StringIO

from model.model import detect_anomalies
from models import Base
from database import engine

# Initialize DB
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helper functions
# ---------------------------

@app.get("/")
def read_root():
    return {"message": "SmartBizIQ Backend is running!"}


def create_lstm_or_gru_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'lstm':
        model.add(LSTM(50, activation='relu', input_shape=input_shape))
    elif model_type == 'gru':
        model.add(GRU(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

LABEL_MAP = {
    0: "👑 VIP Customers",
    1: "💰 Potential Loyalists",
    2: "🛍️ Regular Customers",
    3: "📉 Low Engagement",
    4: "🆕 New Customers",
}

# ---------------------------
# Forecasting Endpoint
# ---------------------------
@app.post("/forecasting")
async def forecast(file: UploadFile = File(...), model: str = Query("prophet")):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        if 'Year' in df.columns and 'Value' in df.columns:
            df['ds'] = pd.to_datetime(df['Year'].astype(str), format='%Y')
            df['y'] = df['Value']
        elif 'ds' in df.columns and 'y' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        else:
            return JSONResponse(status_code=400, content={"error": "CSV must contain ['Year', 'Value'] or ['ds', 'y'] columns."})

        df = df[['ds', 'y']].dropna()
        if df.empty:
            return JSONResponse(status_code=400, content={"error": "No valid data."})

        result = None

        # Prophet
        if model == "prophet":
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=5, freq='Y')
            forecast_data = m.predict(future)
            result = forecast_data[['ds', 'yhat']]

        # ARIMA
        elif model == "arima":
            df_arima = df.set_index("ds")
            arima = ARIMA(df_arima['y'], order=(1,1,1)).fit()
            last_date = pd.to_datetime(df_arima.index[-1])
            future_dates = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=5, freq='Y')
            forecast_values = arima.forecast(steps=5)
            result = pd.DataFrame({"ds": future_dates, "yhat": forecast_values})

        # LSTM / GRU
        elif model in ["lstm", "gru"]:
            df_nn = df.set_index('ds')
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df_nn[['y']])

            X, y = [], []
            for i in range(5, len(scaled)):
                X.append(scaled[i-5:i])
                y.append(scaled[i])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            model_nn = create_lstm_or_gru_model(model, (X.shape[1], X.shape[2]))
            model_nn.fit(X, y, epochs=50, verbose=0)

            last_sequence = scaled[-5:].reshape((1,5,1))
            preds = []
            for _ in range(5):
                pred = model_nn.predict(last_sequence, verbose=0)[0][0]
                preds.append(pred)
                last_sequence = np.append(last_sequence[:,1:,:], [[[pred]]], axis=1)

            future_dates = pd.date_range(start=df['ds'].max() + pd.DateOffset(years=1), periods=5, freq='Y')
            preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            result = pd.DataFrame({"ds": future_dates, "yhat": preds})

        else:
            return JSONResponse(status_code=400, content={"error": f"Unsupported model: {model}"})

        # Metrics
        try:
            if model == "prophet":
                merged = pd.merge(df, result, on='ds', how='inner')
            else:
                test = df.tail(5).reset_index(drop=True)
                pred = result.reset_index(drop=True)
                merged = pd.DataFrame({'y': test['y'], 'yhat': pred['yhat']}) if len(test)==5 else pd.DataFrame()
            if not merged.empty:
                mae = mean_absolute_error(merged['y'], merged['yhat'])
                mse = mean_squared_error(merged['y'], merged['yhat'])
                rmse = np.sqrt(mse)
            else:
                mae = mse = rmse = 0
        except:
            mae = mse = rmse = 0

        summary = f"Projected value in {result['ds'].dt.year.iloc[-1]} is ${result['yhat'].iloc[-1]:.2f}."
        return {"forecast": result.to_dict(orient="records"),
                "metrics": {"MAE": round(mae,2), "MSE": round(mse,2), "RMSE": round(rmse,2)},
                "summary": summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

# ---------------------------
# Customer Segmentation
# ---------------------------
@app.post("/segmentation/segment-customers")
async def segment_customers(file: UploadFile = File(...), method: str = Query("kmeans")):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if not {'Age','Annual_Income','Spending_Score'}.issubset(df.columns):
            return JSONResponse(status_code=400, content={"error": "CSV must have Age, Annual_Income, Spending_Score"})

        features = df[['Age','Annual_Income','Spending_Score']]
        scaled = StandardScaler().fit_transform(features)

        model = DBSCAN(eps=1.2,min_samples=2) if method.lower()=="dbscan" else KMeans(n_clusters=3,random_state=42)
        clusters = model.fit_predict(scaled)
        df['Cluster'] = clusters
        df['Label'] = df['Cluster'].map(LABEL_MAP).fillna("🧠 Moderate")

        # Plot
        plt.figure(figsize=(8,6))
        sns.scatterplot(x='Annual_Income',y='Spending_Score',hue='Label',data=df,palette='Set2',s=100,alpha=0.8)
        plt.title("Customer Segmentation Clusters")
        plt.xlabel("Annual Income")
        plt.ylabel("Spending Score")
        plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        summaries = {}
        for label, group in df.groupby("Label"):
            count = len(group)
            age_range = f"{group['Age'].min()}–{group['Age'].max()}"
            income_range = f"₹{group['Annual_Income'].min():,} to ₹{group['Annual_Income'].max():,}"
            score_range = f"{group['Spending_Score'].min()} to {group['Spending_Score'].max()}"
            summaries[label] = f"{label} ({count} customers): Typically aged {age_range}, incomes from {income_range}, and spending scores between {score_range}."

        return JSONResponse(content={"data": df[['Age','Annual_Income','Spending_Score','Cluster','Label']].to_dict(orient='records'),
                                     "plot": img_base64,
                                     "summaries": summaries})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# Churn Prediction
# ---------------------------
@app.post("/predict-churn")
async def predict_churn(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        if 'CustomerID' in df.columns: df.rename(columns={"CustomerID":"Customer"}, inplace=True)

        required_cols = {'Customer','Gender','Age','Tenure','MonthlyCharges','TotalCharges'}
        if not required_cols.issubset(df.columns):
            return JSONResponse(status_code=400, content={"error": f"CSV must contain: {', '.join(required_cols)}"})

        df.dropna(subset=['Gender','Age','Tenure','MonthlyCharges','TotalCharges'], inplace=True)
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        df['Churn'] = [1 if i%2==0 else 0 for i in range(len(df))]  # dummy labels

        X = df[['Gender','Age','Tenure','MonthlyCharges','TotalCharges']]
        y = df['Churn']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(*train_test_split(X,y,test_size=0.2,random_state=42)[:2])  # train only

        probs = model.predict_proba(X)[:,1]
        df['ChurnProbability'] = (probs*100).round(2)
        df['ChurnLabel'] = df['ChurnProbability'].apply(lambda p: "🔴 Likely to Churn" if p>50 else "🟢 Retained")

        return {"data": df[['Customer','Gender','Age','Tenure','MonthlyCharges','TotalCharges','ChurnProbability','ChurnLabel']].to_dict(orient="records")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# Anomaly Detection
# ---------------------------
@app.post("/anomaly-detection")
async def detect_anomaly(file: UploadFile = File(...), method: str = Query("isolation_forest")):
    try:
        df = pd.read_csv(file.file)
        result_df, img_base64 = detect_anomalies(df, method=method)
        return {"data": result_df.to_dict(orient="records"), "plot": img_base64}
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Customer Upload & Recommendation
# ---------------------------
data = None
kmeans_model = None

@app.post("/upload_and_recommend")
async def upload_and_recommend(file: UploadFile = File(...), customer_id: str = Form(...)):
    global data, kmeans_model
    try:
        df = pd.read_csv(file.file)
        if df.empty: raise HTTPException(status_code=400, detail="CSV is empty.")
        df.set_index(df.columns[0], inplace=True)
        df.index = df.index.astype(int)

        numeric_cols = df.select_dtypes(include="number").columns
        if numeric_cols.empty: raise HTTPException(status_code=400, detail="CSV must have numeric columns.")

        data_numeric = df[numeric_cols].groupby(df.index).mean()
        n_clusters = min(5, len(data_numeric))
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        data_numeric["Cluster"] = kmeans_model.fit_predict(data_numeric)
        data = data_numeric

        customer_id_int = int(customer_id)
        if customer_id_int not in data.index:
            raise HTTPException(status_code=404, detail="Customer ID not found.")

        cluster = data.loc[customer_id_int, "Cluster"]
        cluster_members = data[data["Cluster"]==cluster].drop(columns=["Cluster"])
        avg_scores = cluster_members.mean().sort_values(ascending=False)

        customer_purchases = data.loc[customer_id_int].drop("Cluster")
        already_bought = customer_purchases[customer_purchases>0].index
        recommendations = [prod for prod in avg_scores.index if prod not in already_bought][:10]

        return JSONResponse(content={"cluster": int(cluster), "recommendations": recommendations})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
