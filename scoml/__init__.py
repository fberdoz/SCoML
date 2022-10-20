import yaml
import argparse
from argparse import Namespace
from scoml.constants import SCOML_DEFAULT_CONFIG_FILE

# create local logger
import logging
logger = logging.getLogger(__name__)


def get_args():
    """Get the command line and file arguments."""
    
    # add arguments
    parser = argparse.ArgumentParser(description="SCoML")
    parser.add_argument("--config-file", "-c",
                        help="yaml configuration file", 
                        type=str, 
                        dest = "config_file",
                        default=SCOML_DEFAULT_CONFIG_FILE)
  
    # parse command line arguments and logging for info
    line_args, unknown = parser.parse_known_args()
    logger.info("Line arguments: {}".format(", ".join(["{}='{}'".format(k, v) for k, v in line_args.__dict__.items()])))    
    for value in unknown:
        logger.warning("Unknown line argument '{}'".format(value))
        
    # read config file
    with open(line_args.config_file, mode='r') as f:
        dict_args = yaml.load(f, Loader=yaml.FullLoader)
        
    # log file arguments
    logger.info("File arguments: {}".format(dict_args))

    # merge line and file arguments (priority to command line arguments)
    if dict_args is not None:
        dict_args.update(vars(line_args))
        args = dict_args
    else:
        logger.warning("Empty config file")
        args = vars(line_args)
    
    return args

