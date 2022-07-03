import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models


def load_model(feature_dim, code_length, num_classes,):

    resnet34 = models.resnet34(pretrained=True)
    model = Resnet34(resnet34, feature_dim, code_length, num_classes)

    return model


class Resnet34(nn.Module):
    def __init__(self, origin_model, feature_dim, code_length=64, num_classes=100):
        super(Resnet34, self).__init__()
        self.code_length = code_length
        self.features = nn.Sequential(*list(origin_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_1 = nn.Linear(512, feature_dim)
        self.fc_2 = nn.Linear(512, feature_dim)
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
        self.hash_layer = nn.Linear(feature_dim, code_length)
        self.classifier = nn.Linear(code_length, num_classes)
        self.apply(_weights_init)
        
        dict_fc = self.fc_1.state_dict()
        self.fc_2.load_state_dict(dict_fc)

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)
        x = F.relu(x.view(x.size(0), -1))
        x1 = self.fc_1(x)
        x2 = self.fc_2(x)

        concept_selector1 = torch.tanh(x1)
        concept_selector2 = torch.tanh(x2)

        alpha = concept_selector2 * concept_selector1

        x1 = x1 * alpha
        x2 = x2 * alpha
        x = torch.cat((x1, x2), dim=1)
        
        x = self.fc(x)

        direct_feature = x

        x = torch.tanh(x)

        hash_codes = self.hash_layer(x)

        hash_codes = torch.tanh(hash_codes)

        assignments = self.classifier(hash_codes)

        return hash_codes, assignments, direct_feature


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
