import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tf_template import Model
from tf_template.utils import get_dataloaders, test, train


# Model Parameters
LEARNING_RATE = 0.01
EPOCHS = 101
BATCH_SIZE = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Run details
TEST = True  # Just sets the seed, might need to add a flag
LOAD = False
SAVE = False

# Path names
RUN_PATH = ""
LOAD_PATH = ""  # No logging or save path

# Configuring logging
logging.basicConfig(
    filename="runs/example.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=logging.INFO,
)

# Logging hyperparameters
logging.info(
    f"""
Learning Rate: {LEARNING_RATE}
Epochs: {EPOCHS}
Batch Size: {BATCH_SIZE}
Device: {DEVICE}
"""
)

# Tensorboard writer
writer = SummaryWriter()


def main() -> None:
    # Set seed if in test mode
    if TEST:
        torch.manual_seed(24)

    # Initialising model, loss function and optimizier
    model = Model(1, 1).to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    logging.info(
        f"""
Model:\n{model}\n
Loss function: {loss_fn}\n
Optimizer:\n{optimizer}
"""
    )

    # Get split data loaders
    train_dataloader, test_dataloader = get_dataloaders(BATCH_SIZE)

    logging.info(f"Number of train batches: {len(train_dataloader)}")
    logging.info(f"Number of test batches: {len(test_dataloader)}")

    # Load model params
    if LOAD:
        # TODO: Change to correct path
        model.load_state_dict(torch.load("model.pth"))
        logging.info(f"Loaded model from {'p'}")

    # Training loop
    for epoch in tqdm(range(EPOCHS)):
        logging.debug(f"Epoch: {epoch}")
        train(train_dataloader, DEVICE, optimizer, model, loss_fn)
        if epoch % 10 == 0:
            test(test_dataloader, DEVICE, model, loss_fn)

    # Save model params
    if SAVE:
        # TODO: Change to correct path
        torch.save(model.state_dict(), "model.pth")
        logging.info(f"Saved model to {'p'}")

    # Needs to be done at the end to save the final loss
    writer.add_hparams(
        {"Learning Rate": LEARNING_RATE, "Batch Size": BATCH_SIZE, "Epochs": EPOCHS},
        # TODO loss needs to update - architecture change
        {"Loss": 5},
    )

    # Closing tensorboard write
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
