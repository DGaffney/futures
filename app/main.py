import os
from datetime import datetime, timedelta
from typing import Generator
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
import sqlalchemy
from sqlalchemy.orm import Session
from app.stock_predictor import StockPredictor
from app.models import Symbol, Prediction, SessionLocal

app = FastAPI()
FUTURE_PROJECTION_WINDOW = 3*60 #Three hours into future
HISTORICAL_DATA_WINDOW=14 #Amount of historical data to review for prediction (bigger is better but slower)
def to_dict(obj):
    return {c.key: getattr(obj, c.key) for c in sqlalchemy.inspect(obj).mapper.column_attrs}

def get_db() -> Generator:
    """
    Dependency provider for database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_symbol_by_name(db: Session, symbol_name: str) -> Symbol:
    return db.query(Symbol).filter(Symbol.name == symbol_name.upper()).first()

def get_latest_prediction_by_symbol(db: Session, symbol_id: int) -> Prediction:
    return db.query(Prediction).filter(Prediction.symbol_id == symbol_id).order_by(Prediction.prediction_start_time.desc()).first()

@app.get("/symbols")
def get_all_symbols(db: Session = Depends(get_db)) -> dict:
    """
    Fetches all tracked stock symbols.
    """
    symbols = db.query(Symbol).all()
    return {"symbols": [to_dict(symbol) for symbol in symbols]}

@app.post("/symbol/{symbol_name}")
def start_tracking(symbol_name: str, db: Session = Depends(get_db)) -> dict:
    """
    Starts tracking a given stock symbol.
    """
    db_symbol = get_symbol_by_name(db, symbol_name)
    if db_symbol:
        raise HTTPException(status_code=400, detail="Symbol already tracked")
    symbol = Symbol(name=symbol_name.upper())
    db.add(symbol)
    db.commit()
    return {"message": f"Started tracking {symbol_name}"}

@app.delete("/symbol/{symbol_name}")
def stop_tracking(symbol_name: str, db: Session = Depends(get_db)) -> dict:
    """
    Stops tracking a given stock symbol.
    """
    symbol = get_symbol_by_name(db, symbol_name)
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol not found")
    db.delete(symbol)
    db.commit()
    return {"message": f"Stopped tracking {symbol_name}"}

@app.get("/symbol/ping/{symbol_name}")
def ping(background_tasks: BackgroundTasks, symbol_name: str, db: Session = Depends(get_db)):
    """
    Triggers a new prediction for a given stock symbol and saves it to the database.
    """
    symbol = get_symbol_by_name(db, symbol_name)
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol not found")
    
    predictor = StockPredictor(os.getenv("POLYGON_API_KEY"), symbol_name.upper(), (datetime.now() - timedelta(days=HISTORICAL_DATA_WINDOW)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'), 900000)

    # Adding task to background tasks
    for aspect in ["o", "c", "h", "l"]:
        background_tasks.add_task(predictor.prepare_predict_and_save, db, symbol, aspect, FUTURE_PROJECTION_WINDOW)
    
    return {"message": f"Prediction is being saved for {symbol_name.upper()}"}

@app.get("/symbol/{symbol_name}/{aspect}")
def get_prediction(symbol_name: str, aspect: str, db: Session = Depends(get_db)) -> dict:
    """
    Fetches the latest prediction for a given stock symbol.
    """
    symbol = get_symbol_by_name(db, symbol_name)
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol not found")
    prediction = get_latest_prediction_by_symbol(db, symbol.id)
    if not prediction:
        raise HTTPException(status_code=400, detail="No prediction found")
    return {"prediction": prediction.prediction_value, "symbol": symbol_name, "aspect": aspect}

