# Credit: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

# Training XGBoost for linear regression
import logging
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import xgboost as xgb

from torch_template.utils import generate_data


# Constants
SEED = 1
TEST_SIZE = 0.20

# Getting time for path
t = time.localtime()
TIME = time.strftime("%H-%M %d_%m_%y", t)

# Path names
PATH = "/home/jovyan/runs"

if os.path.exists(PATH) == False:
    os.mkdir(PATH)

OUTPUT_PATH = os.path.join(PATH, TIME)
os.mkdir(OUTPUT_PATH)

# Configuring logging
logging.basicConfig(
    filename=os.path.join(OUTPUT_PATH, "run.log"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=logging.DEBUG,
)


def main():
    # Instantiate model
    model = xgb.XGBRegressor()
    logging.debug(model)
    # TODO: Add functionality for loading model

    # generate data
    X, Y = generate_data(320)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=SEED
    )

    # fit model
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)

    # evaluate predictions
    logging.info(f"MSE: {mse(y_test, y_pred)}")

    # TODO: Add functionality for saving model


if __name__ == "__main__":
    main()
