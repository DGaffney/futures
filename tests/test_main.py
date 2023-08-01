import os
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, Mock, patch
from sqlalchemy.orm import Session

# Assuming you have these somewhere
from app.main import app, get_db
from app.models import Base, engine, Symbol, Prediction


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

# Mock database session for dependency injection
@pytest.fixture
def mock_db_session():
    return Mock(spec=Session)


def test_start_tracking(client, mocker):
    rand_val = str(np.random.random())
    mocker.patch('app.main.get_symbol_by_name', return_value=None)
    response = client.post(f"/symbol/test{rand_val}")
    assert response.status_code == 200
    assert response.json() == {"message": f"Started tracking test{rand_val}"}

def test_start_tracking_already_exists(client, mocker):
    symbol = Symbol(name='test')
    mocker.patch('app.main.get_symbol_by_name', return_value=symbol)
    response = client.post("/symbol/test")
    assert response.status_code == 400
    assert response.json() == {"detail": "Symbol already tracked"}

def test_stop_tracking(client, mocker):
    rand_val = str(np.random.random())
    response = client.post(f"/symbol/test{rand_val}")
    response = client.delete(f"/symbol/test{rand_val}")
    assert response.status_code == 200
    assert response.json() == {"message": f"Stopped tracking test{rand_val}"}

def test_stop_tracking_not_found(client, mocker):
    mocker.patch('app.main.get_symbol_by_name', return_value=None)
    response = client.delete("/symbol/test")
    assert response.status_code == 400
    assert response.json() == {"detail": "Symbol not found"}

def test_get_prediction_symbol_not_found(client, mocker):
    mocker.patch('app.main.get_symbol_by_name', return_value=None)
    response = client.get("/symbol/test")
    assert response.status_code == 400
    assert response.json() == {"detail": "Symbol not found"}

def test_get_prediction(client, mocker):
    symbol = Symbol(name='test')
    mocker.patch('app.main.get_symbol_by_name', return_value=symbol)
    prediction = Prediction(symbol_id=1, prediction_value='10')
    mocker.patch('app.main.get_latest_prediction_by_symbol', return_value=prediction)
    response = client.get("/symbol/test")
    assert response.status_code == 200
    assert response.json() == {"prediction": '10'}

def test_get_prediction_no_prediction(client, mocker):
    symbol = Symbol(name='test')
    mocker.patch('app.main.get_symbol_by_name', return_value=symbol)
    mocker.patch('app.main.get_latest_prediction_by_symbol', return_value=None)
    response = client.get("/symbol/test")
    assert response.status_code == 400
    assert response.json() == {"detail": "No prediction found"}

def test_ping(client, mocker):
    symbol = Symbol(name='test')
    mocker.patch('app.main.get_symbol_by_name', return_value=symbol)
    mocker.patch('app.main.StockPredictor')
    response = client.get("/symbol/ping/test")
    assert response.status_code == 200
    assert response.json() == {"message": "Prediction is being saved for test"}

def test_ping_symbol_not_found(client, mocker):
    symbol = Symbol(name='test')
    mocker.patch('app.main.get_symbol_by_name', return_value=None)
    mocker.patch('app.main.StockPredictor')
    response = client.get("/symbol/ping/test")
    assert response.status_code == 400
    assert response.json() == {"detail": "Symbol not found"}
