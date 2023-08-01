import json
from typing import List, Callable, Dict, Union
from datetime import datetime, timedelta, time
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
from statsforecast.models import DynamicOptimizedTheta
import requests
from sqlalchemy.orm import Session
import pandas as pd
from .models import Prediction, Symbol

REPORTED_CONFIDENCE_INTERVALS = [75, 90, 95, 99]

class StockPredictor:
    """Predicts future stock prices using the StatsForecast model and Polygon API."""

    def __init__(self, 
                 api_key: str,
                 ticker_symbol: str, 
                 start_date: str, 
                 end_date: str, 
                 limit: int, 
                 get_request: Callable[[str], requests.Response] = requests.get) -> None:
        """
        Initializes the StockPredictor object.
        
        Args:
            api_key (str): The Polygon API key.
            ticker_symbol (str): The stock symbol to predict.
            start_date (str): The start date for the historical data.
            end_date (str): The end date for the historical data.
            limit (int): The maximum number of data points to retrieve.
            get_request (Callable[[str], requests.Response]): The function to perform HTTP GET requests.
        """
        self.api_key = api_key
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.limit = limit
        self.url = self.api_url()
        self.model = None
        self.get_request = get_request

    def api_url(self) -> str:
        """Constructs the URL for the Polygon API call.
        
        Returns:
            str: The URL for the Polygon API call.
        """
        return f"https://api.polygon.io/v2/aggs/ticker/{self.ticker_symbol}/range/1/minute/{self.start_date}/{self.end_date}?adjusted=true&sort=asc&limit={self.limit}&apiKey={self.api_key}"

    def get_response(self) -> Dict[str, any]:
        """Performs an HTTP GET request to fetch data from the Polygon API.

        Returns:
            Dict[str, any]: A dictionary containing the JSON response data.
        """
        response = self.get_request(self.url)
        return response.json()

    def response_to_df(self, data: Dict[str, any], column: str) -> pd.DataFrame:
        """Converts the JSON response data into a pandas DataFrame.
    
        Args:
            data (Dict[str, any]): The JSON response data.
            column (str): The timeseries value of interest.

        Returns:
            pd.DataFrame: A DataFrame containing the historical data.
        """
        df = pd.DataFrame(data['results'])
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df = df[['t', column]]
        df.columns = ['ds', 'y']
        df['unique_id'] = column
        return df.sort_values(by="ds", ascending=True)


    def interpolate_data(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Interpolates missing data in the DataFrame, with a threshold of 60 minutes for gaps.
        Use as quick and dirty solution to not interpolate between trading days.

        Args:
            df (pd.DataFrame): The original DataFrame.
            column (str): The column to fill in the DataFrame.

        Returns:
            pd.DataFrame: The interpolated DataFrame.
        """
        # Ensure 'ds' is of datetime type
        df['ds'] = pd.to_datetime(df['ds'])
        # Set 'ds' as index
        df.set_index('ds', inplace=True)
        # Resample data to every minute
        df = df.resample('1T').asfreq()
        # Find where the gaps in the time index are <= 60 minutes
        diff = df.index.to_series().diff()
        mask = ((diff <= pd.Timedelta(minutes=60)) | (diff.isnull()))
        # Interpolate only where mask is True
        df['y'] = df['y'].where(mask).interpolate(method='linear')
        # Reset index
        df.reset_index(inplace=True)
        df['unique_id'] = column
        return df

    def replace_time_with_timestep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces time data in the DataFrame with an auto-incrementing timestep.
        Resolves issue of stocks having large temporal gaps by timestamps.

        Args:
            df (pd.DataFrame): The original DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with the 'ds' column replaced by timesteps.
        """
        df['ds'] = range(len(df))
        return df

    def strip_na(self, df: pd.DataFrame, column: str = 'y') -> pd.DataFrame:
        """
        Strips rows with NaN values in a specified column from the DataFrame.

        Args:
            df (pd.DataFrame): The original DataFrame.
            column (str): The column to check for NaN values. Defaults to 'y'.

        Returns:
            pd.DataFrame: The DataFrame with rows containing NaN in the specified column removed.
        """
        return df.dropna(subset=[column]).reset_index(drop=True)

    def fetch_data(self, column: str) -> pd.DataFrame:
        """Fetches historical data for the stock from the Polygon API.

        Args:
            column (str): The column to predict (e.g. 'c' for closing value at that time slice)

        Returns:
            pd.DataFrame: A DataFrame containing the historical data.
        """
        data = self.get_response()
        return self.strip_na(self.replace_time_with_timestep(self.interpolate_data(self.response_to_df(data, column), column)))

    def train_model(self, df: pd.DataFrame) -> StatsForecast:
        """Trains a StatsForecast model on the historical data.
        
        Args:
            df (pd.DataFrame): A DataFrame containing the historical data.
        
        Returns:
            StatsForecast: The trained model.
        """
        model = StatsForecast(
            models = [DynamicOptimizedTheta(season_length=60)],
            freq = 'T'
        )
        model.fit(df)
        return model

    def predict(self, model: StatsForecast, periods: int) -> pd.DataFrame:
        """Predicts future stock prices using the trained model.
        
        Args:
            model (StatsForecast): The trained model.
            periods (int): The number of future periods to predict.
        
        Returns:
            pd.DataFrame: A DataFrame containing the predicted data.
        """
        forecast = model.predict(h=periods, level=REPORTED_CONFIDENCE_INTERVALS)
        return forecast

    def get_existing_prediction(self, db: Session, start_time: datetime, end_time: datetime, column: str, symbol_id: int) -> Union[Prediction, None]:
        return db.query(Prediction).filter_by(
            prediction_start_time=start_time,
            prediction_end_time=end_time,
            symbol_aspect=column,
            symbol_id=symbol_id,
        ).first()

    def store_prediction(self, db: Session, symbol: Symbol, prediction: Prediction, start_time: datetime, end_time: datetime, column: str, safe_value: Dict) -> None:
        if prediction is None:
            prediction = Prediction(
                prediction_start_time=start_time,
                prediction_end_time=end_time,
                symbol_aspect=column,
                prediction_value=safe_value
            )
            symbol.predictions.append(prediction)
            db.add(prediction)
            db.commit()
        else:
            prediction.safe_value = safe_value
            db.commit()
        
    def add_predictions(self, db: Session, symbol: Symbol, column: str, forecast: pd.DataFrame) -> Dict:
        """Adds the predicted data to the database.
        
        Args:
            db (Session): The SQLAlchemy session.
            symbol (Symbol): The stock symbol object.
            forecast (pd.DataFrame): A DataFrame containing the predicted data.
        """
        #force as to_json first to kill any complex float datatypes that would otherwise throw errors when saving...
        safe_value = json.loads(json.dumps({
            "timesteps": forecast['ds'].tolist(),
            "values": forecast['DynamicOptimizedTheta'].tolist(),
            "lower": forecast['DynamicOptimizedTheta-lo-99'].tolist(),
            "upper": forecast['DynamicOptimizedTheta-hi-99'].tolist()
        }))
        start_time = datetime.combine(datetime.now().date(), time())
        end_time = start_time+timedelta(minutes=len(safe_value["timesteps"])),
        prediction = self.get_existing_prediction(db, start_time, end_time, column, symbol.id)
        # If it does not exist, create a new one
        self.store_prediction(db, symbol, prediction, start_time, end_time, column, safe_value)
        return safe_value

    def predict_and_save(self, db: Session, symbol: Symbol, column: str, model: StatsForecast, periods: int) -> pd.DataFrame:
        """Predicts future stock prices and saves them to the database.
        
        Args:
            db (Session): The SQLAlchemy session.
            symbol (Symbol): The stock symbol object.
            model (StatsForecast): The trained model.
            periods (int): The number of future periods to predict.
        
        Returns:
            pd.DataFrame: A DataFrame containing the predicted data.
        """
        forecast = self.predict(model, periods)
        self.add_predictions(db, symbol, column, forecast)
        return forecast

    def prepare_predict_and_save(self, db: Session, symbol: Symbol, column: str, periods: int) -> None:
        """Predicts future stock prices and saves them to the database.
        
        Args:
            db (Session): The SQLAlchemy session.
            symbol (Symbol): The stock symbol object.
            column (str): The symbol measurement to forecast ('o', 'h', 'c', or 'l')
            periods (int): The number of future periods to predict.
        
        Returns:
            None
        """
        df = self.fetch_data(column)
        model = self.train_model(df)
        return self.predict_and_save(db, symbol, column, model, periods)