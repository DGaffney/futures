import pytest
from unittest import mock
from datetime import datetime
from pandas.testing import assert_frame_equal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .main import StockPredictor
from .models import Prediction, Symbol

@pytest.fixture
def mock_get_request_success():
    """A mock function for successful get_request."""

    def _mock_get_request_success(url: str):
        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "results": [{"t": 1590585600000, "c": 123.45}, {"t": 1590672000000, "c": 125.67}]
        }
        return mock_response

    return _mock_get_request_success

@pytest.fixture
def mock_get_request_failure():
    """A mock function for unsuccessful get_request."""

    def _mock_get_request_failure(url: str):
        mock_response = mock.Mock()
        mock_response.json.return_value = {"error": "API request failed."}
        return mock_response

    return _mock_get_request_failure

@pytest.fixture
def in_memory_db():
    engine = create_engine('sqlite:///:memory:')
    Session = sessionmaker(bind=engine)
    return Session()

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
            'y': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'unique_id': 'c'
        }
    )

def test_api_url(stock_predictor):
    assert stock_predictor.api_url() == "https://api.polygon.io/v2/aggs/ticker/GOOGL/range/1/minute/2020-05-28/2020-06-01?adjusted=true&sort=asc&limit=10&apiKey=fake_api_key"

def test_get_response_success(stock_predictor):
    assert stock_predictor.get_response() == {"results": [{"t": 1590585600000, "c": 123.45}, {"t": 1590672000000, "c": 125.67}]}

def test_get_response_failure(stock_predictor, mock_get_request_failure):
    stock_predictor.get_request = mock_get_request_failure
    with pytest.raises(Exception) as e:
        stock_predictor.get_response()
    assert str(e.value) == "API request failed."

def test_response_to_df(stock_predictor):
    data = {"results": [{"t": 1590585600000, "c": 123.45}, {"t": 1590672000000, "c": 125.67}]}
    df = stock_predictor.response_to_df(data, 'c')
    assert list(df.columns) == ['ds', 'y', 'unique_id']
    assert len(df) == 2

def test_interpolate_data(stock_predictor):
    df = pd.DataFrame({"ds": [datetime(2020, 1, 1, 12), datetime(2020, 1, 1, 12, 1), datetime(2020, 1, 1, 12, 3)], "y": [1, np.nan, 3]})
    df = stock_predictor.interpolate_data(df, 'y')
    assert df['y'].tolist() == [1.0, 2.0, 3.0]

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
    assert len(df) == 2


@patch('requests.get')
def test_train_model(mock_get, stock_predictor):
    mock_get.return_value.json.return_value = create_sample_data()
    df = predictor.fetch_data('c')
    model = predictor.train_model(df)
    assert isinstance(model, StatsForecast)

@patch('requests.get')
def test_predict(mock_get, stock_predictor):
    mock_get.return_value.json.return_value = create_sample_data()
    df = predictor.fetch_data('c')
    model = predictor.train_model(df)
    forecast = predictor.predict(model, 10)
    assert isinstance(forecast, pd.DataFrame)

def test_add_predictions(stock_predictor):
    session = Session()
    symbol = Mock()
    forecast = create_sample_data()
    predictor.add_predictions(session, symbol, forecast)
    assert len(symbol.predictions) == 10

@patch('requests.get')
def test_predict_and_save(mock_get, stock_predictor):
    mock_get.return_value.json.return_value = create_sample_data()
    session = Session()
    symbol = Mock()
    df = predictor.fetch_data('c')
    model = predictor.train_model(df)
    forecast = predictor.predict_and_save(session, symbol, model, 10)
    assert isinstance(forecast, pd.DataFrame)
    assert len(symbol.predictions) == 10

def test_predict_and_save(stock_predictor, in_memory_db):
    symbol = Symbol(name="GOOGL")
    in_memory_db.add(symbol)
    in_memory_db.commit()

    model = stock_predictor.train_model(df=stock_predictor.fetch_data('c'))
    stock_predictor.predict_and_save(in_memory_db, symbol, model, periods=10)

    assert in_memory_db.query(Prediction).count() == 10
