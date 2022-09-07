import scoml
from scoml.utils import load_data

if __name__ == "__main__":
    
    # Initialize module and parse arguments
    args = scoml.init()
    
    # Load data
    dataset = scoml.utils.load_data(args)
    
    # Initialize network and distribute data across clients
    network = Network(args)
    network.split_data(args, dataset)
    
    # Simulator
    simulator = Simulator(args, network)
    
    # Simulation
    simulator.run(args)
    
    # Outputs
    simulator.generate_outputs(args)
    