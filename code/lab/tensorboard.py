import torch
from tensorboardX import SummaryWriter
writer = SummaryWriter('./data/tensorboard_logs')

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(10)

# Call flush() method to make sure that all pending events have been written to disk.
writer.flush()

# close writter
writer.close()


# run python tensorboard.py
# then run tensorboard --logdir=./data/tensorboard_logs