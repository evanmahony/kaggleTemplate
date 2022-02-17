import logging
import os

from flask import Flask, request
import pandas as pd
import torch

from torch_template import Model

PATH = "/home/jovyan"
LOAD_PATH = os.path.join(PATH, "runs/03-56 17_02_22/model.pth")
OUTPUT_PATH = os.path.join(PATH, "runs/api")

# Configuring logging
logging.basicConfig(
    filename=os.path.join(OUTPUT_PATH, "run.log"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=logging.INFO,
)

app = Flask(__name__)

model = Model(1, 1).to("cpu")
logging.info(f"Model:\n{model}")

model.load_state_dict(torch.load(LOAD_PATH))
logging.info(f"Loaded model from {LOAD_PATH}")
model.eval()
X = torch.tensor([1]).type(torch.LongTensor).to("cpu")
logging.info(f"X: {X.type}")
logging.info(f"{model.forward(X)}")


@app.route("/model")
def model():
    logging.info(f"Model:\n{model}")
    return f"Model:\n{model}"


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_json = request.get_json()
        input_df = pd.read_json(input_json)
        logging.info(f"Input DataFrame: {input_df}\n")
        X = torch.tensor(input_df.values)[0]
        logging.info(f"X shape: {X.shape}\n")
        logging.info(f"X: {X}\n")
        pred = model.forward(X)
        return pred


if __name__ == "__main__":
    app.run(host="localhost", port=6006)
