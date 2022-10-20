import os
import logging
LOG_FILE = "./log/tmp/debug.log"

def init_logger(name=None):
    """Create a logger.
    
    Args:
        -name: name of logger (max 11 characters)
        -log_filepath: filepath where to create the logfile (will overwrite previous logfile)
        """

    # clear debug.log file
    if os.path.isfile(LOG_FILE):
        os.remove(LOG_FILE)
    
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
    fh_formatter = logging.Formatter('%(asctime)-25s| %(name)-16s| %(levelname)-8s| %(message)s')
    ch_formatter = logging.Formatter('%(name)-16s| %(levelname)-8s| %(message)s')
    fh.setFormatter(fh_formatter)
    ch.setFormatter(ch_formatter)
    
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger