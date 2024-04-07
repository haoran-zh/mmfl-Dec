import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np

def evaluation(model, data, batch_size, device, args):
    model.eval()
    # split data into validation set (first 30%) and test set (last 70%)
    # Calculate the number of validation samples
    num_val_samples = int(len(data) * 0.3)
    indices = list(range(len(data)))

    # Shuffle indices if you want to randomize the validation set
    # random.shuffle(indices)

    # Select the indices for the validation and test sets
    val_indices = indices[:num_val_samples]
    test_indices = indices[num_val_samples:]

    if args is None:
        # do nothing, use all data
        pass
    else:
        if args.validation is True:
            data = Subset(data, val_indices)
        else:
            data = Subset(data, test_indices)
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


def get_local_loss(task_number, num_clients, task_type, type_iid, tasks_data_info, tasks_data_idx, global_models, device, batch_size):
    localLoss = np.zeros((task_number, num_clients))
    for cl in range(num_clients):
        for task in range(task_number):
            if type_iid[task] == 'iid' or task_type[task] == 'shakespeare':
                client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][
                    cl])  # or iid_partition depending on your choice
            elif type_iid[task] == 'noniid':
                client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][0][
                    cl])  # or iid_partition depending on your choice
            accu, loss = evaluation(model=global_models[task], data=client_data,
                                    batch_size=batch_size, device=device, args=None)  # use all data
            # accu = localAccResults[task, cl, round-1]
            # localLossResults[task, cl, round] = loss
            # localAcc[task, cl] = accu
            localLoss[task, cl] = loss
    return localLoss