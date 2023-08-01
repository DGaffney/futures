import pytest
from unittest import mock
from unittest.mock import Mock, patch
from datetime import datetime
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from app.stock_predictor import StockPredictor
from statsforecast import StatsForecast
from app.models import Prediction, Symbol

RESPONSE_FIXTURE = {"ticker":"AAPL","queryCount":10,"resultsCount":10,"adjusted":True,"results":[{"v":4428,"vw":195.7709,"o":195.89,"c":195.53,"h":195.9,"l":195.53,"t":1690790400000,"n":121},{"v":674,"vw":195.6834,"o":195.67,"c":195.7,"h":195.7,"l":195.67,"t":1690790460000,"n":44},{"v":1869,"vw":195.5604,"o":195.7,"c":195.39,"h":195.7,"l":195.39,"t":1690790520000,"n":78},{"v":671,"vw":195.5228,"o":195.54,"c":195.5,"h":195.54,"l":195.5,"t":1690790580000,"n":20},{"v":682,"vw":195.5727,"o":195.59,"c":195.59,"h":195.59,"l":195.59,"t":1690790700000,"n":27},{"v":1003,"vw":195.6769,"o":195.69,"c":195.69,"h":195.69,"l":195.69,"t":1690790820000,"n":33},{"v":1305,"vw":195.6092,"o":195.59,"c":195.59,"h":195.59,"l":195.59,"t":1690791120000,"n":36},{"v":111,"vw":195.7384,"o":195.74,"c":195.74,"h":195.74,"l":195.74,"t":1690791240000,"n":3},{"v":515,"vw":195.6124,"o":195.61,"c":195.61,"h":195.61,"l":195.61,"t":1690791360000,"n":33},{"v":166,"vw":195.61,"o":195.61,"c":195.61,"h":195.61,"l":195.61,"t":1690791420000,"n":2}],"status":"OK","request_id":"7fea49fbee5d3f40e6a66076f1f4be4e","count":10,"next_url":"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/minute/1690791480000/2023-07-31?cursor=bGltaXQ9MTAmc29ydD1hc2M"}
@pytest.fixture
def mock_get_request_success():
    """A mock function for successful get_request."""

    def _mock_get_request_success(url: str):
        mock_response = mock.Mock()
        mock_response.json.return_value = RESPONSE_FIXTURE
        return mock_response

    return _mock_get_request_success

@pytest.fixture
def stock_predictor(mock_get_request_success):
    return StockPredictor(
        api_key="fake_api_key",
        ticker_symbol="GOOGL",
        start_date="2020-05-28",
        end_date="2020-06-01",
        limit=10,
        get_request=mock_get_request_success
    )


def create_sample_data():
    return pd.DataFrame(
        data={
            'ds': range(10),
            'DynamicOptimizedTheta': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'DynamicOptimizedTheta-lo-99': [8, 18, 28, 38, 48, 58, 68, 78, 88, 98],
            'DynamicOptimizedTheta-hi-99': [12, 22, 32, 42, 52, 62, 72, 82, 92, 102],
            'unique_id': 'c'
        }
    )

def test_api_url(stock_predictor):
    assert stock_predictor.api_url() == "https://api.polygon.io/v2/aggs/ticker/GOOGL/range/1/minute/2020-05-28/2020-06-01?adjusted=true&sort=asc&limit=10&apiKey=fake_api_key"

def test_get_response_success(stock_predictor):
    assert stock_predictor.get_response() == RESPONSE_FIXTURE

def test_response_to_df(stock_predictor):
    data = {"results": [{"t": 1590585600000, "c": 123.45}, {"t": 1590672000000, "c": 125.67}]}
    df = stock_predictor.response_to_df(data, 'c')
    assert list(df.columns) == ['ds', 'y', 'unique_id']
    assert len(df) == 2

def test_interpolate_data(stock_predictor):
    df = pd.DataFrame({"ds": [datetime(2020, 1, 1, 12), datetime(2020, 1, 1, 12, 1), datetime(2020, 1, 1, 12, 3)], "y": [1, np.nan, 3]})
    df = stock_predictor.interpolate_data(df, 'y')
    assert df['y'].tolist() == [1.0, 1.6666666666666665, 2.333333333333333, 3.0]

def test_replace_time_with_timestep(stock_predictor):
    df = pd.DataFrame({"ds": [datetime(2020, 1, 1, 12), datetime(2020, 1, 1, 12, 1), datetime(2020, 1, 1, 12, 3)], "y": [1, 2, 3]})
    df = stock_predictor.replace_time_with_timestep(df)
    assert df['ds'].tolist() == [0, 1, 2]

def test_strip_na(stock_predictor):
    df = pd.DataFrame({"ds": [0, 1, 2], "y": [1, np.nan, 3]})
    df = stock_predictor.strip_na(df, 'y')
    assert df['y'].tolist() == [1.0, 3.0]

def test_fetch_data(stock_predictor):
    df = stock_predictor.fetch_data('c')
    assert list(df.columns) == ['ds', 'y', 'unique_id']


@patch('requests.get')
def test_train_model(mock_get, stock_predictor):
    mock_get.return_value.json.return_value = create_sample_data()
    df = stock_predictor.fetch_data('c')
    model = stock_predictor.train_model(df)
    assert isinstance(model, StatsForecast)

@patch('requests.get')
def test_predict(mock_get, stock_predictor):
    mock_get.return_value.json.return_value = create_sample_data()
    df = stock_predictor.fetch_data('c')
    model = stock_predictor.train_model(df)
    forecast = stock_predictor.predict(model, 10)
    assert isinstance(forecast, pd.DataFrame)

def test_add_predictions(stock_predictor, mocker):
    symbol = Mock()
    session = Session()
    forecast = create_sample_data()
    mocker.patch('app.stock_predictor.StockPredictor.get_existing_prediction', return_value=None)
    mocker.patch('app.stock_predictor.StockPredictor.store_prediction', return_value=None)
    forecast = stock_predictor.add_predictions(session, symbol, 'c', forecast)
    assert isinstance(forecast, dict)

@patch('requests.get')
def test_predict_and_save(mock_get, stock_predictor, mocker):
    mock_get.return_value.json.return_value = create_sample_data()
    session = Session()
    symbol = Mock()
    mocker.patch('app.stock_predictor.StockPredictor.get_existing_prediction', return_value=None)
    mocker.patch('app.stock_predictor.StockPredictor.store_prediction', return_value=None)
    df = stock_predictor.fetch_data('c')
    model = stock_predictor.train_model(df)
    forecast = stock_predictor.predict_and_save(session, symbol, 'c', model, 10)
    assert isinstance(forecast, pd.DataFrame)
