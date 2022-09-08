from scoml.client import Client

# create local logger
import logging
logger = logging.getLogger(__name__)


class Network():
    """Store all the clients and the information related to the network"""

    def __init__(self, args):
        self.args = args
        self.clients = []
        self.last_given_ip = -1
        self.ip_list = []

        for client_config, n in args.client_types.items():
            for i in range(n):
                self.add_client(client_config)

        logger.debug("Network initialized")
    
    def add_client(self, config_file):
        """Add a new client to the network and manage its id"""
        
        # create new client
        new_ip = self.last_given_ip + 1
        self.last_given_ip += 1
        new_client = Client(config_file, new_ip)

        # check that the feature dim of the new client is >= the shared feature dim
        if new_client.feature_dim < self.args.shared_dim:
            logger.warning(("Client {} has 'feature_dim' < 'shared_dim'."
                            "'feature_dim' is thus set to 'shared_dim' ({}) for compatibility in SCoML (config: {})."
                            ).format(new_ip, self.args.shared_dim, config_file))
            new_client.feature_dim = self.args.shared_dim
        
        # Add client to network
        self.clients.append(new_client)
        self.ip_list.append(new_ip)

    def delete_client(self, client_id):
        """Delete a client from the network and manage its id"""
        raise NotImplementedError