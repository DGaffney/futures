# Stock Price Predictor

The Stock Price Predictor is a machine learning application that predicts future stock prices. It uses the StatsForecast model to train and predict stock prices based on historical data fetched from the Polygon API. This prediction model is hosted as a microservice using Docker and can be accessed via HTTP endpoints.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

To run this project, you will need:

- Docker: You can download it from [here](https://www.docker.com/products/docker-desktop)
- Python 3.10+: You can download it from [here](https://www.python.org/downloads/)

### Installation

1. Clone this repository:
    ```
    git clone https://github.com/your-repo/stock-price-predictor.git
    ```
2. Build the Docker images:
    ```
    docker-compose build
    ```
3. Start the containers:
    ```
    docker-compose up
    ```
4. To stop the containers, use:
    ```
    docker-compose down
    ```

## Application Architecture

This application primarily consists of a `StockPredictor` class that interacts with the Polygon API to fetch stock price data, trains the StatsForecast model on this data, and makes future price predictions.

### StockPredictor Class

The `StockPredictor` class is responsible for fetching historical stock price data, training a StatsForecast model, making future price predictions, and storing these predictions in a database. The detailed methods are:

- `__init__(self, api_key: str, ticker_symbol: str, start_date: str, end_date: str, limit: int, get_request = requests.get)`: Initializes the StockPredictor object.
- `api_url(self) -> str`: Constructs the URL for the Polygon API call.
- `get_response(self) -> Dict[str, any]`: Performs an HTTP GET request to fetch data from the Polygon API.
- `response_to_df(self, data: Dict[str, any]) -> pd.DataFrame`: Converts the JSON response data into a pandas DataFrame.
- `fetch_data(self) -> pd.DataFrame`: Fetches historical data for the stock from the Polygon API.
- `train_model(self, df: pd.DataFrame) -> StatsForecast`: Trains a StatsForecast model on the historical data.
- `predict(self, model: StatsForecast, periods: int) -> pd.DataFrame`: Predicts future stock prices using the trained model.
- `add_predictions(self, session: Session, symbol: 'Symbol', forecast: pd.DataFrame) -> None`: Adds the predicted data to the database.
- `predict_and_save(self, session: Session, symbol: 'Symbol', model: StatsForecast, periods: int) -> pd.DataFrame`: Predicts future stock prices and saves them to the database.

### HTTP Endpoints

The application exposes the following HTTP endpoints:

- `/predict/<ticker_symbol>/<start_date>/<end_date>/<limit>`: GET request that returns the predicted future stock prices for the provided ticker symbol. The `start_date` and `end_date` parameters specify the range of historical data to use for model training. The `limit` parameter specifies the maximum number of data points to retrieve.

- `/history/<ticker_symbol>`: GET request that returns the historical stock price data for the provided ticker symbol.

## Testing

To be added: instructions on running unit and integration tests.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- StatsForecast for providing an easy-to-use machine learning model for time series prediction.
- Polygon.io for providing the API for historical stock price data.
