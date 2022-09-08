from scoml.network import Network

# create local logger
import logging
logger = logging.getLogger(__name__)

class SCoMLSimulator():
    
    def __init__(self, args):
        self.args = args
        self.network = Network(args)
        logger.debug("Simulator initialized")

    def init_network(self):
        for client_config, n in self.args.client_types.items():
            for i in range(n):
                self.network.add_client(client_config)


    def train_clients(self):    
        pass

    def communicate(self):
        pass

    def run(self):
        logger.debug("Starting the simulation")
        for e in range(self.args.rounds):
            
            # Local updates on client side
            self.train_clients()
            
            # Communication between peers
            self.communicate()

        logger.debug("Simulation over")
            