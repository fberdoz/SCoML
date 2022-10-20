from scoml.network import Network

# create local logger
import logging
logger = logging.getLogger(__name__)

class Simulator():
    """Root class for the simulation"""
    
    def __init__(self, config):
        self.config = config

        # network initialization
        self.network = Network(config)

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
            
            # logging
            logger.info("Round {} done.".format(r))

        logger.debug("Simulation over")
            