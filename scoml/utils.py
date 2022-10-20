import yaml
import argparse
from argparse import Namespace
from scoml.constants import *

# AI
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets

# create local logger
import logging
logger = logging.getLogger(__name__)


def get_optimizer(opt_type, model, lr):
    """Return an optimizer"""
    if opt_type == SCOML_ADAM_OPT:
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == SCOML_SGD_OPT:
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_type == SCOML_ADAGRAD_OPT:
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        logger.error("Unknown type of optimizer '{}'".format(opt_type))

def get_criterion(loss_type):
    """Return an criterion module (loss)."""
    if loss_type == SCOML_CROSSENTROPY_LOSS:
        return nn.CrossEntropyLoss()
    else:
        logger.error("Unknown type of loss '{}'".format(loss_type))

def load_data(dataset, data_dir="./data", reduced=False, normalize="image-wise", flatten=False):
    """Load the specified dataset.
    
    Arguments:
        - dataset: Name of the dataset to load.
        - data_dir: Directory where to store (or load if already stored) the dataset.
        - reduced: Boolean/'small'/'tiny'/float between 0 and 1. Reduce the dataset size.
        - normalize: Whether to normalize the data ('image-wise' or 'channel-wise').
        - flatten: Whether to flatten the data (i.g. for FC nets).
        
    Returns:
        - train_input: A tensor with the train inputs.
        - train_target: A tensor with the train targets.
        - test_input: A tensor with the test inputs.
        - test_target: A tensor with the test targets.
        - meta: A dictionry with useful metadata on the dataset.
    """
    
    # Initialize meta data:
    meta = {"n_class": None,
            "in_dimension": None}
    
    if dataset == SCOML_CIFAR10_DS:
        # Load
        logger.info("Using CIFAR10 dataset")
        logger.info("Load train data...")
        train_set = datasets.CIFAR10(data_dir + '/cifar10/', train=True,download=True)
        logger.info("Load test data...")
        test_set = datasets.CIFAR10(data_dir + '/cifar10/', train=False,download=True)

        # Process train data
        train_input = torch.from_numpy(train_set.data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float() / 255
        train_target = torch.tensor(train_set.targets, dtype=torch.int64)
        
        # Process test data
        test_input = torch.from_numpy(test_set.data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float() / 255
        test_target = torch.tensor(test_set.targets, dtype=torch.int64)
        
        # Update metadata
        meta["n_class"] = 10
        meta["class_names"] = ["airplane", "automobile", "bird", "cat", 
                               "deer", "dog", "frog", "horse", "ship", "truck"]
    
    elif dataset == SCOML_CIFAR100_DS:
        # Load
        logger.info("Using CIFAR100 dataset")
        logger.info("Load train data...")
        train_set = datasets.CIFAR100(data_dir + '/cifar100/', train=True,download=True)
        logger.info("Load test data...")
        test_set = datasets.CIFAR100(data_dir + '/cifar100/', train=False,download=True)

        # Process train data
        train_input = torch.from_numpy(train_set.data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float() / 255
        train_target = torch.tensor(train_set.targets, dtype=torch.int64)
        
        # Process test data
        test_input = torch.from_numpy(test_set.data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float() / 255
        test_target = torch.tensor(test_set.targets, dtype=torch.int64)
        
        # Update metadata
        meta["n_class"] = 100
        meta["class_names"] = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
                               "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly",
                               "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee",
                               "clock", "cloud", "cockroach", "couch", "cra", "crocodile", "cup", "dinosaur",
                               "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
                               "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard",
                               "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
                               "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck",
                               "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon",
                               "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
                               "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
                               "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
                               "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
                               "woman", "worm"]
        
    elif dataset == SCOML_MNIST_DS:
        logger.info("Using MNIST dataset")
        logger.info("Load train data...")
        train_set = datasets.MNIST(data_dir, train=True, download=True)
        logger.info("Load test data...")
        test_set = datasets.MNIST(data_dir, train=False, download=True)

        # Process train data
        train_input = train_set.data.view(-1, 1, 28, 28).float()
        train_target = train_set.targets
        
        # Process test data
        test_input = test_set.data.view(-1, 1, 28, 28).float()
        test_target = test_set.targets
        
        # Update metadata
        meta["n_class"] = 10
        meta["class_names"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
    elif dataset == SCOML_FMNIST_DS:
        logger.info("Using FMNIST dataset")
        logger.info("Load train data...")
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True)
        logger.info("Load test data...")
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True)

        # Process train data
        train_input = train_set.data.view(-1, 1, 28, 28).float()
        train_target = train_set.targets
        
        # Process test data
        test_input = test_set.data.view(-1, 1, 28, 28).float()
        test_target = test_set.targets
        
        # Update metadata
        meta["n_class"] = 10
        meta["class_names"] = ["T-shirt/top", "Trouser", "Pullover", "Dress", 
                               "Coat", "Sandal", "Shirt", "Sneaker", "Bag",  "Ankle boot"]
    elif dataset == SCOML_EMNIST_DS:
        logger.info("Using EMNIST dataset")
        logger.info("Load train data...")
        train_set = datasets.EMNIST(data_dir, split="balanced", train=True, download=True)
        logger.info("Load test data...")
        test_set = datasets.EMNIST(data_dir, split="balanced", train=False, download=True)

        # Process train data
        train_input = train_set.data.view(-1, 1, 28, 28).permute(0, 1, 3, 2).float()
        train_target = train_set.targets
        
        # Process test data
        test_input = test_set.data.view(-1, 1, 28, 28).permute(0, 1, 3, 2).float()
        test_target = test_set.targets
        
        # Update metadata
        meta["n_class"] = 1
        meta["class_names"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", 
                               "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", 
                               "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "d", 
                               "e", "f", "g", "h", "n", "q", "r", "t"]
    else:
        logger.error("Unknown dataset '{}'.".format(dataset))
    
    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)

    if reduced == "small" or reduced is True:
        train_input = train_input.narrow(0, 0, 2000)
        train_target = train_target.narrow(0, 0, 2000)
        test_input = test_input.narrow(0, 0, 1000)
        test_target = test_target.narrow(0, 0, 1000)
    
    elif reduced == "tiny":
        train_input = train_input.narrow(0, 0, 400)
        train_target = train_target.narrow(0, 0, 400)
        test_input = test_input.narrow(0, 0, 200)
        test_target = test_target.narrow(0, 0, 200)
    
    elif isinstance(reduced, float) and reduced > 0 and reduced < 1.0:
        n_tr = int(reduced * train_input.shape[0])
        train_input = train_input.narrow(0, 0, n_tr)
        train_target = train_target.narrow(0, 0, n_tr)
        
        #n_te = int(reduced * test_input.shape[0])
        #test_input = test_input.narrow(0, 0, n_te)
        #test_target = test_target.narrow(0, 0, n_te)
    
    # Print dataset information
    memory_train = (train_input.element_size() * train_input.nelement() + train_target.element_size() * train_target.nelement())/ 1e6
    memory_test = (test_input.element_size() * test_input.nelement() + test_target.element_size() * test_target.nelement())/ 1e6
    logger.info("Train dataset size: {} ({} MB)".format(tuple(train_input.shape), memory_train))
    logger.info("Test dataset size: {} ({} MB)".format(tuple(test_input.shape), memory_test))

    # Normalization
    if normalize == "channel-wise":
        # Normalize each channels independently
        dims = [i for i in range(train_input.dim()) if i != 1]
        mu = train_input.mean(dim=dims, keepdim=True)
        sig = train_input.std(dim=dims, keepdim=True)   
    
    elif normalize == "image-wise":
        # Normalize all channels
        mu = train_input.mean()
        sig = train_input.std()

    else:
        mu = 0
        sig = 1
    
    # Subtract mean and divide by std
    train_input.sub_(mu).div_(sig)
    test_input.sub_(mu).div_(sig)
    meta["mu"] = mu
    meta["sig"] = sig
    
    # Update metadata
    meta["in_dimension"] = train_input.shape[1:]
    meta["n_train"] = train_input.shape[0]
    meta["n_test"] = test_input.shape[0]
    
    return train_input, train_target, test_input, test_target, meta

class LocalDataset(Dataset):
    """Custom dataset wrapper for local datasets."""
    def __init__(self, inputs, targets):
        """Constructor.
        
        Arguments:
            - inputs: A tensor contaiing the inputs aligned in the 0th dimension.
            - targets: A tensor contaiing the targets aligned in the 0th dimension.
        """
        super(LocalDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets
        self.len = inputs.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]