from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ForecastResult(Base):
    __tablename__ = "forecast_results"

    id = Column(Integer, primary_key=True, index=True)
    model_used = Column(String)
    forecast_year = Column(Integer)
    forecast_value = Column(Float)
    mae = Column(Float)
    mse = Column(Float)
    rmse = Column(Float)
    summary = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
