import yaml

# create local logger
import logging
logger = logging.getLogger(__name__)

class Client():
    """Class containing all the informations of a client (local)"""
    def __init__(self, config_file, ip):
        # metadata
        self.config_file = config_file 
        self.ip = ip

        # read config file
        with open(config_file, mode='r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # store client parameters in class instance
        self.model = config["model"]
        self.epochs_per_round = config["epoch_per_round"]
        self.optimizer = config["optimizer"]
        self.lr = config["lr"]
        self.lambda_kd = config["lambda_kd"]
        self.lambda_disc = config["lambda_disc"]
        self.feature_dim = config["feature_dim"]

        # Log
        logger.debug("Client {} initialized ({})".format(ip, config_file))
        
    def local_update(self):
        """Perform one local update"""
        for e in range(self.epochs_per_rounds):
            pass