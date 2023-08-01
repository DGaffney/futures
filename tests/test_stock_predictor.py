from unittest.mock import Mock, patch
from stock_predictor import StockPredictor
import pandas as pd
from datetime import datetime
from neuralprophet import NeuralProphet

# sample data for mocking
mocked_response = {
    'results': [
        {'t': 1625605400000, 'c': 123.45},
        {'t': 1625606300000, 'c': 125.45},
        {'t': 1625607200000, 'c': 124.45}
    ]
}

mocked_df = pd.DataFrame(mocked_response['results'])
mocked_df['t'] = pd.to_datetime(mocked_df['t'], unit='ms')
mocked_df = mocked_df[['t', 'c']]
mocked_df.columns = ['ds', 'y']

# A mock NeuralProphet model
mocked_model = Mock(spec=NeuralProphet)

# unit tests
def test_api_url():
    predictor = StockPredictor("API_KEY", "TEST", '2023-07-18', '2023-07-31', 1200)
    expected_url = "https://api.polygon.io/v2/aggs/ticker/TEST/range/1/minute/2023-07-18/2023-07-31?adjusted=true&sort=asc&limit=1200&apiKey=API_KEY"
    assert predictor.api_url() == expected_url

@patch('requests.get')
def test_get_response(mock_get):
    mock_get.return_value.json.return_value = mocked_response
    predictor = StockPredictor("API_KEY", "TEST", '2023-07-18', '2023-07-31', 1200)
    assert predictor.get_response() == mocked_response

def test_response_to_df():
    predictor = StockPredictor("API_KEY", "TEST", '2023-07-18', '2023-07-31', 1200)
    assert predictor.response_to_df(mocked_response).equals(mocked_df)

@patch('requests.get')
def test_fetch_data(mock_get):
    mock_get.return_value.json.return_value = mocked_response
    predictor = StockPredictor("API_KEY", "TEST", '2023-07-18', '2023-07-31', 1200)
    assert predictor.fetch_data().equals(mocked_df)

@patch.object(NeuralProphet, "fit")
def test_train_model(mock_fit):
    predictor = StockPredictor("API_KEY", "TEST", '2023-07-18', '2023-07-31', 1200)
    mock_fit.return_value = mocked_model
    assert isinstance(predictor.train_model(mocked_df), NeuralProphet)

@patch.object(NeuralProphet, "make_future_dataframe")
@patch.object(NeuralProphet, "predict")
def test_predict(mock_predict, mock_make_future_dataframe):
    predictor = StockPredictor("API_KEY", "TEST", '2023-07-18', '2023-07-31', 1200)
    mock_forecast = Mock()
    mock_predict.return_value = mock_forecast
    assert predictor.predict(mocked_model, 1) is mock_forecast

@patch.object(StockPredictor, "add_predictions")
@patch.object(StockPredictor, "predict")
def test_predict_and_save(mock_predict, mock_add_predictions):
    predictor = StockPredictor("API_KEY", "TEST", '2023-07-18', '2023-07-31', 1200)
    mock_session = Mock()
    mock_symbol = Mock()
    mock_forecast = Mock()
    mock_predict.return_value = mock_forecast
    assert predictor.predict_and_save(mock_session, mock_symbol, mocked_model, 1) is mock_forecast
    mock_add_predictions.assert_called_once_with(mock_session, mock_symbol, mock_forecast)

@patch.object(StockPredictor, "predict_and_save")
@patch.object(StockPredictor, "train_model")
@patch.object(StockPredictor, "fetch_data")
def test_prepare_predict_and_save(mock_fetch_data, mock_train_model, mock_predict_and_save):
    predictor = StockPredictor("API_KEY", "TEST", '2023-07-18', '2023-07-31', 1200)
    mock_fetch_data.return_value = mocked_df
    mock_train_model.return_value = mocked_model
    predictor.prepare_predict_and_save(Mock(), Mock(), 1)
    mock_predict_and_save.assert_called_once()
