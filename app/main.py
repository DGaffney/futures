from typing import Generator
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from sqlalchemy.orm import Session
from app.stock_predictor import StockPredictor
from app.models import Symbol, Prediction, SessionLocal

app = FastAPI()

def get_db() -> Generator:
    """
    Dependency provider for database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/symbol/{symbol_name}")
def start_tracking(symbol_name: str, db: Session = Depends(get_db)) -> dict:
    """
    Starts tracking a given stock symbol.
    """
    db_symbol = db.query(Symbol).filter(Symbol.name == symbol_name).first()
    if db_symbol:
        raise HTTPException(status_code=400, detail="Symbol already tracked")
    symbol = Symbol(name=symbol_name)
    db.add(symbol)
    db.commit()
    return {"message": f"Started tracking {symbol_name}"}

@app.delete("/symbol/{symbol_name}")
def stop_tracking(symbol_name: str, db: Session = Depends(get_db)) -> dict:
    """
    Stops tracking a given stock symbol.
    """
    symbol = db.query(Symbol).filter(Symbol.name == symbol_name).first()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol not found")
    db.delete(symbol)
    db.commit()
    return {"message": f"Stopped tracking {symbol_name}"}

@app.get("/symbol/{symbol_name}")
def get_prediction(symbol_name: str, db: Session = Depends(get_db)) -> dict:
    """
    Fetches the latest prediction for a given stock symbol.
    """
    symbol = db.query(Symbol).filter(Symbol.name == symbol_name).first()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol not found")
    prediction = db.query(Prediction).filter(Prediction.symbol_id == symbol.id).order_by(Prediction.prediction_time.desc()).first()
    if not prediction:
        raise HTTPException(status_code=400, detail="No prediction found")
    return {"prediction": prediction.prediction_value}

@app.get("/symbol/ping/{symbol_name}")
def ping(background_tasks: BackgroundTasks, symbol_name: str, db: Session = Depends(get_db)):
    """
    Triggers a new prediction for a given stock symbol and saves it to the database.
    """
    symbol = db.query(Symbol).filter(Symbol.name == symbol_name).first()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol not found")
    
    predictor = StockPredictor(os.getenv("POLYGON_API_KEY"), symbol_name, '2023-07-18', '2023-07-31', 1200)

    # Adding task to background tasks
    background_tasks.add_task(predictor.prepare_predict_and_save, db, symbol, 3*60)
    
    return {"message": f"Prediction is being saved for {symbol_name}"}
