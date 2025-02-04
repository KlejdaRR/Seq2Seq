import torch

# Function to train the model for one epoch
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(iterator)

# Function to evaluate the model on a validation/test set
def evaluate_model(model, iterator, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inp_data, target in iterator:
            inp_data, target = inp_data.to(device), target.to(device)
            output = model(inp_data, target, teacher_force_ratio=0)  # No teacher forcing during evaluation
            output = output[:, 1:].reshape(-1, output.shape[2])
            target = target[:, 1:].reshape(-1)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(iterator)

# Function to save a model checkpoint
def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Function to load a model checkpoint
def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
