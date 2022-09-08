import torch
import torch.nn as nn
import torch.nn.functional as F


class _SCoMLModel(nn.Module):
    """Parent class for all model participating in a SCoML framework."""

    def __init__(self):
        super(_SCoMLModel, self).__init__()

class LeNet5(nn.Module):
    """Classical CNN based on LeNet5 architecture."""
    def __init__(self, in_channels, feat_dim, output_shape, dropout=0):
        """Constructor.
    
        Arguments:
            -in_channels: Number of input channels
            -feat_dim: Feature (last hidden layer) dimension.
            -output_shape: Number of class (for the output dimention)
            -dropout: Percentage of neurons to drop.
        """
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels=6, kernel_size=5, stride=1, padding=2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Dropout2d(dropout),
                                      nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Dropout2d(dropout),
                                      nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                      nn.Flatten(start_dim=1),
                                      nn.Linear(120, feat_dim),
                                      nn.Tanh(),
                                      nn.Dropout(dropout))
        self.classifier = nn.Linear(feat_dim, output_shape)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x