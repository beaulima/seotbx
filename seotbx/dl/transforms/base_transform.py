import logging
logger = logging.getLogger(__name__)
import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod


class BaseTransform(ABC):

    def __init__(self, config):
        self.config = None