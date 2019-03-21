import torch
import torch.nn as nn
import horovod as hvd
from utils import get_params
def train_distributed(model, train_loader, optimizer,config,writer):
    nums_epochs = config["train_params"]["num_epochs"]
    loss_func = nn.MSELoss()
    for i in range(nums_epochs):
        sum_loss = 0
        n_iter = len(train_loader)
        for batch_idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            prediction = model(X)
            loss = loss_func(y, prediction)
            sum_loss += loss.data
            loss.backward()
            optimizer.step()
            if hvd.rank() == 0:
                writer.add_scalar('Train/Loss', loss.data, n_iter * i + batch_idx)
        if hvd.rank()==0:
            writer.add_scalar('Train/Epoch_Loss', sum_loss / n_iter, i)

def train(model, train_loader, optimizer,config,writer):
    nums_epochs = config["train_params"]["num_epochs"]
    loss_func = nn.MSELoss()
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    device = get_params(config["train_params"],"device", device)
    for i in range(nums_epochs):
        sum_loss = 0
        n_iter = len(train_loader)
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(X)
            loss = loss_func(y, prediction)
            sum_loss += loss.data
            loss.backward()
            optimizer.step()
            writer.add_scalar('Train/Loss', loss.data, n_iter * i + batch_idx)
        writer.add_scalar('Train/Epoch_Loss', sum_loss / n_iter, i)