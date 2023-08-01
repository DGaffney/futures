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
    git clone https://github.com/DGaffney/futures.git
    ```
2. Build the Docker images:
    ```
    docker-compose build
    ```
3. Start the containers:
    ```
    POLYGON_API_KEY=XXX docker-compose up #specify the key otherwise you're not going to get real data...
    ```
4. To stop the containers, use:
    ```
    docker-compose down
    ```

## Application Architecture

This application primarily consists of a `StockPredictor` class that interacts with the Polygon API to fetch stock price data, trains the StatsForecast model on this data, and makes future price predictions.

## StockPredictor Class

This class predicts future stock prices using the StatsForecast model and Polygon API.

### Initialization

During initialization, the following parameters are accepted:

- `api_key (str)`: The Polygon API key.
- `ticker_symbol (str)`: The stock symbol to predict.
- `start_date (str)`: The start date for the historical data.
- `end_date (str)`: The end date for the historical data.
- `limit (int)`: The maximum number of data points to retrieve.
- `get_request (Callable[[str], requests.Response])`: The function to perform HTTP GET requests. It defaults to `requests.get`.

The initialization function constructs the URL for the Polygon API call and initializes an instance of `StatsForecast`.

### Methods

- `api_url()`: Constructs the URL for the Polygon API call.

- `get_response()`: Performs an HTTP GET request to fetch data from the Polygon API.

- `response_to_df(data: Dict[str, any], column: str)`: Converts the JSON response data into a pandas DataFrame.

- `interpolate_data(df: pd.DataFrame, column: str)`: Interpolates missing data in the DataFrame, with a threshold of 60 minutes for gaps.

- `replace_time_with_timestep(df: pd.DataFrame)`: Replaces time data in the DataFrame with an auto-incrementing timestep.

- `strip_na(df: pd.DataFrame, column: str)`: Strips rows with NaN values in a specified column from the DataFrame.

- `fetch_data(column: str)`: Fetches historical data for the stock from the Polygon API.

- `train_model(df: pd.DataFrame)`: Trains a StatsForecast model on the historical data.

- `predict(model: StatsForecast, periods: int)`: Predicts future stock prices using the trained model.

- `add_predictions(session: Session, symbol: Symbol, column: str, forecast: pd.DataFrame)`: Adds the predicted data to the database.

- `predict_and_save(session: Session, symbol: Symbol, column: str, model: StatsForecast, periods: int)`: Predicts future stock prices and saves them to the database.

- `prepare_predict_and_save(session: Session, symbol: Symbol, column: str, periods: int)`: It fetches the data, trains the model, predicts future stock prices, and saves them to the database. It's a wrapper function that combines the other functions.

## HTTP Endpoints

This section provides an overview of the HTTP endpoints that our FastAPI application exposes and how they can be used.

- **GET /symbols**: This endpoint returns a list of all the stock symbols that are currently being tracked. It doesn't require any parameters. The response is a JSON object with a single field "symbols", which is an array of symbol names.

- **POST /symbol/{symbol_name}**: This endpoint starts tracking a new stock symbol. Replace `{symbol_name}` with the name of the stock symbol you want to start tracking. If the symbol is already being tracked, the server will respond with a 400 error. On success, the server responds with a message saying that tracking has started.

- **DELETE /symbol/{symbol_name}**: This endpoint stops tracking a stock symbol. Replace `{symbol_name}` with the name of the stock symbol you want to stop tracking. If the symbol isn't being tracked, the server will respond with a 400 error. On success, the server responds with a message saying that tracking has stopped.

- **GET /symbol/{symbol_name}**: This endpoint returns the latest prediction for a given stock symbol. Replace `{symbol_name}` with the name of the stock symbol you're interested in. If the symbol isn't being tracked, or if there's no prediction available, the server will respond with a 400 error. On success, the server responds with a JSON object containing the prediction value.

- **GET /symbol/ping/{symbol_name}**: This endpoint triggers a new prediction for a given stock symbol and saves it to the database. Replace `{symbol_name}` with the name of the stock symbol you're interested in. If the symbol isn't being tracked, the server will respond with a 400 error. On success, the server responds with a message saying that the prediction is being saved.

These endpoints provide the ability to manage which stock symbols are being tracked and fetch their predictions. Make sure to replace `{symbol_name}` with the actual symbol name when using the `/symbol/{symbol_name}` endpoints.

## Running database upgrades

- Make changes on app/models.py to alter columns / tables as needed,
- `alembic revision --autogenerate -m "Changes as of [DATE]"`
- `alembic upgrade head`

## Testing

`pytest tests` from inside container to run test suite.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- StatsForecast for providing an easy-to-use machine learning model for time series prediction.
- Polygon.io for providing the API for historical stock price data.
