"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    @abstractmethod
    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError


    @abstractmethod
    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError
