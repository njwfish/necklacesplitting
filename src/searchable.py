from abc import ABCMeta, abstractmethod


class LocalSearcheable(metaclass=ABCMeta):
    """Abstract class to be inherited to allow a class to be searched locally."""

    @abstractmethod
    def get_neighbors(self):
        pass

    @abstractmethod
    def gen_score(self):
        pass
