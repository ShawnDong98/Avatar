import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out