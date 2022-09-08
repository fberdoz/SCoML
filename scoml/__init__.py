import os
from scoml.utils import get_args

# create local logger
import logging
logger = logging.getLogger(__name__)



def init():
    """Initilialization at the system-level"""
    
    # get arguments
    args = get_args()

    return args