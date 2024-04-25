'''This module contains custom logger which will be shared by all the modules'''

import logging
from parameters import LOG_FILE_PATH
from util.generic_utilities import check_for_directory


# create a directory for logs if it does not exist
# remove the last element from the string as it is the file name
log_file_dir = LOG_FILE_PATH.rsplit('/', maxsplit=1)[0]
check_for_directory(log_file_dir)

# basic configuration for logging
logging.basicConfig(level=logging.DEBUG, #messages with level DEBUG or higher will be printed
                    format='%(asctime)s - %(levelname)s -  %(message)s', #format of the log message
                    datefmt='%Y-%m-%d %H:%M:%S', #format of the timestamp
                    filename=LOG_FILE_PATH, #name of the log file
                    filemode='a' #append to the log file
                    )

# create a custom logger
logger = logging.getLogger(__name__)
