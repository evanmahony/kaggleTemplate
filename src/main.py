import torch
from tqdm import tqdm

import example

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 256

print(f"Using {DEVICE} device")


def main():
    model = example.Model(1, 1).to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_dataloader, test_dataloader = example.utils.get_dataloaders(BATCH_SIZE)

    load = False
    if load:
        model.load_state_dict(torch.load("model.pth"))

    for epoch in tqdm(range(EPOCHS)):
        print(f"Epoch {epoch}")
        model.train()
        example.utils.train(train_dataloader, DEVICE, optimizer, model, loss_fn)
        if epoch % 10 == 0:
            example.utils.test(test_dataloader, DEVICE, model, loss_fn)

    save = False
    if save:
        torch.save(model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")


if __name__ == "__main__":
    main()
