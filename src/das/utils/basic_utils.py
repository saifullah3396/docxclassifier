
"""
Defines geenral-purpose utility functions for the stack.
"""

import logging
import os
import re
import sys
from enum import Enum

import coloredlogs

if 'LOG_LEVEL' in os.environ:
    LOG_LEVEL = os.environ['LOG_LEVEL']
else:
    LOG_LEVEL = 'INFO'


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of "
            f"{list(cls._value2member_map_.keys())}"
        )


def configure_logger(logger: logging.Logger):
    """
    Creates a logger with given name

    Args:
        name: Logger name
    """

    logger.propagate = False
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    coloredlogs.install(level=LOG_LEVEL, logger=logger)
    return logger


def create_logger(name: str):
    """
    Creates a logger with given name

    Args:
        name: Logger name
    """
    logger = logging.getLogger(name)
    return configure_logger(logger)


class StdoutLogger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
