
from torch.utils.tensorboard import SummaryWriter
from scoml.network import Network
import numpy as np

# create local logger
import logging
logger = logging.getLogger(__name__)

class Simulator():
    """Root class for the simulation"""
    
    def __init__(self, config):
        self.config = config

        # network initialization
        self.network = Network(config)

        # create tensorboard writer for performance tracking
        self.writer = SummaryWriter()
        
        logger.debug("Simulator initialized succesfully")

    def _train_clients(self):
        """Perform local training on each clients device."""
        
        # iterate through clients
        for ip, client in self.network.clients.items():
            client.local_update()

    def _communicate(self):
        pass

    def run(self):
        logger.debug("Starting the simulation")
        for r in range(1, self.config["rounds"]+1):
            
            # Local updates on client side
            self._train_clients()
            
            # Communication between peers
            self._communicate()
            
            # monitoring performance
            self.writer.add_scalar("Test", np.random.random(), r)
            
            # logging
            logger.info("Round {} done.".format(r))

        logger.debug("Simulation over")
            