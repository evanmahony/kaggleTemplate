import logging
import os
import time

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

# Getting time for path
t = time.localtime()
TIME = time.strftime("%H-%M %d_%m_%y", t)

# Path names
PATH = "/home/jovyan/runs"
OUTPUT_PATH = os.path.join(PATH, TIME)
os.mkdir(OUTPUT_PATH)
LOAD_PATH = ""

# Configuring logging
logging.basicConfig(
    filename=os.path.join(OUTPUT_PATH, "run.log"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=logging.INFO,
)

print(f"Log file saved to {os.path.join(OUTPUT_PATH, 'run.log')}")

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
writer = SummaryWriter(OUTPUT_PATH)


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

    logging.info(
        f"""
Number of train batches: {len(train_dataloader)}
Number of test batches: {len(test_dataloader)}
"""
    )
    
    # Load model params
    if LOAD:
        # TODO: Change to correct path
        model.load_state_dict(torch.load(LOAD_PATH))
        logging.info(f"Loaded model from {LOAD_PATH}")

    # Training loop
    for epoch in tqdm(range(EPOCHS)):
        logging.debug(f"Epoch: {epoch}")
        
        loss = train(train_dataloader, DEVICE, optimizer, model, loss_fn)
        logging.debug(f"Train loss = {loss}")
        writer.add_scalar('Loss/train', loss, epoch)
        
        if epoch % 10 == 0:
            loss = test(test_dataloader, DEVICE, model, loss_fn)
            logging.info(f"Test loss = {loss}")
            writer.add_scalar('Loss/test', loss, epoch)
            
    # Save model params
    if SAVE:
        # TODO: Change to correct path
        torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'model.pth'))
        logging.info(f"Saved model to {os.path.join(OUTPUT_PATH, 'model.pth')}")

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
