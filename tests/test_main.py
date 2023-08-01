from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@patch('main.StockPredictor.get_response')
def test_start_tracking(mock_get_response):
    mock_get_response.return_value = {"results": []}  # Mocked data
    response = client.post("/symbol/TEST")
    assert response.status_code == 200
    assert response.json() == {"message": "Started tracking TEST"}

@patch('main.StockPredictor.get_response')
def test_start_tracking_duplicate(mock_get_response):
    mock_get_response.return_value = {"results": []}  # Mocked data
    response = client.post("/symbol/TEST")
    assert response.status_code == 400

@patch('main.StockPredictor.get_response')
def test_stop_tracking(mock_get_response):
    mock_get_response.return_value = {"results": []}  # Mocked data
    response = client.delete("/symbol/TEST")
    assert response.status_code == 200
    assert response.json() == {"message": "Stopped tracking TEST"}

@patch('main.StockPredictor.get_response')
def test_stop_tracking_not_found(mock_get_response):
    mock_get_response.return_value = {"results": []}  # Mocked data
    response = client.delete("/symbol/TEST")
    assert response.status_code == 400

@patch('main.StockPredictor.get_response')
def test_get_prediction_not_found(mock_get_response):
    mock_get_response.return_value = {"results": []}  # Mocked data
    response = client.get("/symbol/TEST")
    assert response.status_code == 400
