import logging
logger = logging.getLogger(__name__)
import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod


class BaseOptimiser(ABC):

    def __init__(self, config):
        self.optimiser = None