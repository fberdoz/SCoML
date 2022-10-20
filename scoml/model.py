import torch
import torch.nn as nn
import torch.nn.functional as F

# create local logger
import logging
logger = logging.getLogger(__name__)

#import constants
from scoml.constants import (SCOML_LENET5_NN,
                            )

def get_representer(client_config, network_args):
    """Return the specified model using the client and network configurations."""
    if client_config["model_type"] == SCOML_LENET5_NN:
        representer = LeNet5(in_channels=network_args["meta"]["in_dimension"][0], feat_dim=client_config["feature_dim"], dropout=client_config["dropout"])
    else:
        logger.error("Unknown model type '{}'. See scoml.constant file to see which models are implemented.".format(client_config["model_type"]))
        
    return representer

def get_classifier(feature_dim, n_class):
    """Return a simple linear layer classifier."""
    
    return nn.Linear(feature_dim, n_class)
    
class LeNet5(nn.Module):
    """Classical CNN based on LeNet5 architecture."""
    def __init__(self, in_channels, feat_dim, dropout=0):
        """Constructor.
    
        Arguments:
            -in_channels: Number of input channels
            -feat_dim: Feature (last hidden layer) dimension.
            -dropout: Percentage of neurons to drop.
        """
        super(LeNet5, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels=6, kernel_size=5, stride=1, padding=2), 
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
        

    def forward(self, x):
        x = self.model(x)
        return x
    
