import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod

class BaseTrainer(ABC):

    def __init__(self, config):
        self.config = None

        self.model = None,
        self.data_loader = None
        self.min_iteration = 0
        self.max_iteration = 0


    @abstractmethod
    def train_one_cycle(self):

        pass


    @abstractmethod
    def train(self):

        pass
