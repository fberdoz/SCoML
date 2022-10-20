import scoml
from scoml.simulator import Simulator

# initialize logger
from logger import init_logger
logger = init_logger()
logger.debug("Root logger initialized")

# Load config
config = scoml.get_args()

# create simulator instance
simulator = Simulator(config)

# run simulation
simulator.run()
