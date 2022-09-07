import os
import argparse

from scoml.constants import LOG_FILE
from scoml.utils import get_logger

def init():
    
    # clear debug log file
    if os.path.isfile(LOG_FILE):
        os.remove(LOG_FILE)
    
    # create local logger
    logger = get_logger(__name__)
    
    # add arguments
    parser = argparse.ArgumentParser(description="SCoML")
    parser.add_argument("--config_file", "-c", 
                        help="yaml configuration file", 
                        type=str, 
                        default="")
  
    # parse arguments and log 
    args, unknown = parser.parse_known_args()
    logger.info("Line arguments: {}".format(args))    
    for value in unknown:
        logger.warning("Unknown argument '{}'".format(value))
        
        
        
    import yaml

    with open(args.config_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)
    
    return args