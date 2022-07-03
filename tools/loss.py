import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, assignments, targets):
        batch_size = assignments.size(0)

        assignments = F.softmax(assignments, dim=1)
        loss = torch.sum(- torch.log(assignments + 1e-6) * targets) / batch_size

        return loss


class CenConLoss(nn.Module):
    def __init__(self):
        super(CenConLoss, self).__init__()
        self.t = 1.0

    def forward(self, hashcode, center, label):
        cos_sim = F.cosine_similarity(hashcode.unsqueeze(1), center.unsqueeze(0), dim=2)

        positives = (torch.exp(cos_sim * self.t) * label)
        denominator = torch.exp(cos_sim * self.t) * (1 - label)
        loss = -torch.log(torch.sum(positives, dim=1) / torch.sum(denominator, dim=1))
        loss = torch.mean(loss)

        return loss
