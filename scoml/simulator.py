class SCoMLSimulator():
    
    def __init__(self, network):
        self.network = network
        
    def run(self, args):
        
        for e in range(args.rounds):
            
            # Local updates on client side
            self.network.train()
            
            # Communication between peers
            self.network.communicate()
            
            # Logging
            self.log()
            