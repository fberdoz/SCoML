import yaml
import argparse
import logging
import datetime


from scoml.constants import LOG_FILE

def get_logger(name):
    """Create a logger.
    
    Args:
        -name: name of logger (max 11 characters)
        """
    
    # create logger with name
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # create file handler which logs even debug messages
    fh = logging.FileHandler(LOG_FILE, mode='a')
    fh.setLevel(logging.DEBUG)
    
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # create formatter and add it to the handlers
    fh_formatter = logging.Formatter('%(asctime)-25s| %(name)-16s| %(levelname)-9s| %(message)s')
    ch_formatter = logging.Formatter('%(name)-16s| %(levelname)-9s| %(message)s')
    fh.setFormatter(fh_formatter)
    ch.setFormatter(ch_formatter)
    
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger



def get_args():
    
    logger = get_logger(__name__)
    logger.info("inside utils")
    
    parser = argparse.ArgumentParser(description="SCoML")
    parser.add_argument("--yaml_config_file", "--cf", 
                        help="yaml configuration file", 
                        type=str, 
                        default="")
    
    args, unknown = parser.parse_known_args()
   
    
    return args

