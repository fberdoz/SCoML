# standard libraries
import numpy as np

# scoml
import scoml
from scoml.client import Client
from scoml.constants import *

# create local logger
import logging
logger = logging.getLogger(__name__)


class Network():
    """Store all the clients and the information related to the network"""

    def __init__(self, config):
        # initialzation
        self.clients = {}
        self.last_given_ip = -1

        # load dataset
        self.train_input, self.train_target, self.test_input, self.test_target, config["meta"] = scoml.utils.load_data(config["dataset"], reduced=config["reduced"])
        
        # create all the clients iteratively
        config["n_clients"] = 0
        self.config = config
        for client_config_file, n in config["client_types"].items():
            self.config["n_clients"] += n
            for i in range(n):
                self._add_client(client_config_file)
        
        
        # parition the data across clients
        self._partition_data()
        
        # confirm network initialization
        logger.debug("Network initialized succesfully")
    
    def _add_client(self, client_config_file):
        """Add a new client to the network and manage its id"""
        
        # create new client
        new_ip = self.last_given_ip + 1
        self.last_given_ip += 1
        new_client = Client(client_config_file, new_ip, self.config)
        
        # Add client to network
        self.clients[new_ip] = new_client
        
    def _partition_data(self):
        """Partition and distribute the data across the clients. Inspired by https://github.com/Xtra-Computing/NIID-Bench"""
        
        # iid split (i.e. uniformly at random across peers)
        if self.config["partition"] == SCOML_IID_SPLIT:
            idxs = np.random.permutation(self.config["meta"]["n_train"])
            idxs_batch = np.array_split(idxs, self.config["n_clients"])
            self.idxs_map = {ip: idxs_batch[i] for i, ip in enumerate(self.clients.keys())}
        
        # non-iid split using a dirichelt distrubtion for the label (using concentration beta)
        elif self.config["partition"] == SCOML_NIID_DIR_SPLIT:
            
            # minimum number of samples per label & per clients
            n_sample_min_threshold = 10
            n_sample_min = 0
            
            # maximium number of iteration to try to partition the data
            iteration = 0
            max_iteration = 1000
            while n_sample_min < n_sample_min_threshold and iteration < max_iteration:
                idxs_batch = [[] for _ in range(self.config["n_clients"])]
                for k in range(self.config["meta"]["n_class"]):
                    
                    # gather indices of samples with label k and shuffle them
                    idx_k = np.where(self.train_target == k)[0]
                    np.random.shuffle(idx_k)
                    
                    # sample a dirichlet distrubtion
                    proportions = np.random.dirichlet(np.repeat(self.config["beta"], self.config["n_clients"]))

                    # balance so that each client gets at most n_train/n_clients total samples
                    proportions = np.array([p * (len(idx_j) < (self.config["meta"]["n_train"] / self.config["n_clients"])) for p, idx_j in zip(proportions, idxs_batch)])
                    proportions = proportions / proportions.sum()
                    
                    # split samples of label k between clients (using np.split)
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idxs_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idxs_batch, np.split(idx_k, proportions))]
                    
                    # check that every client has at least n_sample_min_threshold samples for each class (and that loop iterates at most max_iteration times)
                    n_sample_min = min([len(idx_j) for idx_j in idxs_batch])
                    iteration +=1
                
            # logging in case no partition was found in time
            if iteration == max_iteration:
                logging.error("Unable to partition the data in such a way that every client has at least {} samples per class.".format(n_sample_min_threshold))
            
            # record indices into index map
            self.idxs_map = {ip: np.random.permutation(idxs_batch[i]) for i, ip in enumerate(self.clients.keys())}
        
        # quantity skew using dirichlet distribution
        elif self.config["partition"] == SCOML_SIZE_DIR_SPLIT:
            idxs = np.random.permutation(self.config["meta"]["n_train"])
            
            # minimum number of samples per label & per clients
            n_sample_min_threshold = 10
            n_sample_min = 0
        
            # maximium number of iteration to try to partition the data
            iteration = 0
            max_iteration = 1000
            while n_sample_min < n_sample_min_threshold and iteration < max_iteration:
                proportions = np.random.dirichlet(np.repeat(self.config["beta"], self.config["n_clients"]))
                proportions = proportions/proportions.sum()
                n_sample_min = np.min(proportions*len(idxs))
            proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
            idxs_batch = np.split(idxs, proportions)
            self.idxs_map = {ip: np.random.permutation(idxs_batch[i]) for i, ip in enumerate(self.clients.keys())}
            
            # logging in case no partition was found in time
            if iteration == max_iteration:
                logging.error("Unable to partition the data in such a way that every client has at least {} samples per class.".format(n_sample_min_threshold))
        
        else:
            logging.error("Unkown partition type. Check 'scoml.constants' file to see valid partition options.")
            
        # creating the dataset in each client
        for ip, client in self.clients.items():
            idxs = self.idxs_map[ip]
            client.create_local_dl(self.train_input[idxs], self.train_target[idxs])
            