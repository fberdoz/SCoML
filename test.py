import scoml
from scoml.simulator import SCoMLSimulator

# initialize logger
from logger import init_logger
logger = init_logger()
logger.debug("Root logger initialized")

args = scoml.init()

simulator = SCoMLSimulator(args)

# high level pipeline
simulator.run()
