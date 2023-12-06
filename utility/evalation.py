import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluation(model, data, batch_size, device):
    model.eval()
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    running_loss = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        accuracy = correct/total
        loss = running_loss/total

    return accuracy, loss