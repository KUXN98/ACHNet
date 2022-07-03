import torch
import numpy as np


def generate_code(model, dataloader, code_length, num_classes, device, flag):
    """
    Generate hash code
    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.
    Returns
        code(torch.Tensor): Hash code.
        assignment(torch.Tensor): assignment.
        label(torch.Tensor): label of sample
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        assignment = torch.zeros([N, num_classes])
        label = torch.zeros(N, dtype=torch.long)
        for data, target, index in dataloader:
            # print(index)
            data = data.to(device)
            hash_code, class_assignment, _ = model(data)
            code[index, :] = hash_code.sign().cpu()
            assignment[index, :] = class_assignment[:data.size(0), :].cpu()
            if flag == 0:
                lab = torch.nonzero(target, as_tuple=False)[:, 1]
                label[index] = lab.long().cpu()
    torch.cuda.empty_cache()
    return code, assignment, label


def generate_hash_center(model, dataloader, device, code_length):
    """
    Generate hash_center
    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.
    Returns
        code(torch.Tensor): hash_center.
    """
    model.eval()
    with torch.no_grad():
        hash_center = torch.zeros([100, code_length])
        counter = torch.zeros([100])
        for data, targets, _ in dataloader:
            data, targets = data.to(device), targets.to(device)
            hash_code, _, _ = model(data)
            direct_feature = hash_code.to('cpu')
            index = torch.nonzero(targets, as_tuple=False)[:, 1]
            index = index.to('cpu')
            for j in range(len(data)):
                hash_center[index[j], :] = hash_center[index[j], :] + direct_feature[j, :]
                counter[index[j]] = counter[index[j]] + 1

        for k in range(100):
            hash_center[k, :] = hash_center[k, :] / counter[k]
    torch.cuda.empty_cache()
    return hash_center


def get_correct_num(output, labels):
    """
    Get the number of correctly classified samples
    Args
        output: output of classification layer.
        label: label of sample.
    Returns
        number: number of correctly classified samples.
    """
    return output.argmax(dim=1).eq(labels).sum().item()


# Get the number of correctly classified samples for each class


def sample_num_per_class(output, label, class_num, flag):
    class_sample_num = list(np.zeros([class_num], dtype=int))
    class_correct_num = list(np.zeros([class_num], dtype=int))

    for i in label:
        class_sample_num[i] += 1
    if flag == 0:
        output = output.argmax(axis=1)
    for i in range(len(label)):
        if output[i] == label[i]:
            class_correct_num[label[i]] += 1
    return class_correct_num, class_sample_num


# Calculate the classification accuracy for each class


def correct_per_class(class_correct_num, class_sample_num, class_num):
    accuracy_per_class = list(np.zeros(class_num))
    for i in range(class_num):
        if class_sample_num[i] != 0:
            accuracy_per_class[i] = round(100 * class_correct_num[i] / class_sample_num[i], 3)
    return accuracy_per_class
