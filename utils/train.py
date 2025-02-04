import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

def train_model(model, iterator, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inp_data, target in iterator:
        inp_data, target = inp_data.to(device), target.to(device)
        output = model(inp_data, target)
        output = output[:, 1:].reshape(-1, output.shape[2])
        target = target[:, 1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(iterator)

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:  # Add this condition
        optimizer.load_state_dict(checkpoint["optimizer"])
