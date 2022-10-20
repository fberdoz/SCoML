import yaml

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# custom models
import scoml.model as mdl
from scoml.utils import get_optimizer, get_criterion, LocalDataset

# create local logger
import logging
logger = logging.getLogger(__name__)

class Client():
    """Class containing all the informations of a client (local)"""
    
    def __init__(self, config_file, ip, args):
        # metadata
        self.config_file = config_file 
        self.ip = ip

        # read config file and store dictionary in class instance
        with open(config_file, mode='r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # check that the feature dim of the new client is >= the shared feature dim
        if config["feature_dim"] < args["shared_dim"]:
            logger.warning(("Client {} has 'feature_dim' ({}) smaller than 'shared_dim' ({})."
                            "'feature_dim' is thus set to 'shared_dim' for compatibility in SCoML (config: {})."
                            ).format(ip, config["feature_dim"], args["shared_dim"], config_file))
            config["feature_dim"] = args["shared_dim"]        
        
        # store client parameters in class instance
        self.config = config
        self.representer = mdl.get_representer(config, args)
        self.classifier = mdl.get_classifier(config["feature_dim"], args["meta"]["n_class"])
        self.model = nn.Sequential(self.representer, self.classifier)
        self.optimizer = get_optimizer(config["optimizer"], self.model, config["lr"])
        self.criterion = get_criterion(config["loss"])
        
        
        # data loaders
        self.train_dl = None

        # log
        logger.debug("Client {} initialized succesfully".format(ip))
    
    def create_local_dl(self, x, y):
        """Create a local dataset and a dataloader using the input data x and the target data y."""
        
        self.ds = LocalDataset(x, y)
        self.dl = DataLoader(self.ds, batch_size=self.config["batch_size"], shuffle=True)
        
    
    def local_update(self):
        """Perform one local update"""
        
        # ensure training mode
        self.model.train()
        
        # train for the given number of epochs
        for e in range(self.config["epochs_per_round"]):
            
            # iterate over batches
            for inputs, targets in self.dl:
                self.optimizer.zero_grad()
                
                # Local forward pass
                features = self.representer(inputs)
                logits = self.classifier(features)
                        
                # Optimization step
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()