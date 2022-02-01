# Credit: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

# Training XGBoost for linear regression
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import xgboost as xgb

from example import utils


# Constants
SEED = 1
TEST_SIZE = 0.20

# Configuring logging
logging.basicConfig(
    filename="runs/example.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=logging.DEBUG,
)


def main():
    # generate data
    X, Y = utils.generate_data(320)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=SEED
    )

    # fit model no training data
    model = xgb.XGBRegressor()
    logging.debug(model)
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)

    # evaluate predictions
    logging.info(f"MSE: {mse(y_test, y_pred)}")


if __name__ == "__main__":
    main()
