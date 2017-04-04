import time
import numpy as np
from src.searchable import LocalSearcheable


class LocalSearch:
    """
    Implements a simple local search function for any problem that can be solved by searching through states, 
    optimizing a search surface for some score
    """
    def __init__(self, obj):
        """Ensures the object is searchable
        :param obj: a LocalSearcheable object to be searched
        """
        if isinstance(obj, LocalSearcheable):
            self.obj = obj

    def search(self, stop):
        """Search through the states and neighboring states, moving the the next state according to a probability
        distribution heavily favoring the highest scoring neighbors. Often the search steps backwards, exploring
        different paths, but it tracks the best object to ensure that is what is returned after the allotted search 
        time.
        :param stop: number of seconds for which to run the search
        :return: the best scoring object found
        """
        t_end = time.time() + stop
        best_solution = self.obj
        best_score = best_solution.score
        while time.time() < t_end:
            # Get all the neigboring states of the object in question and sort by the score
            neighbors = self.obj.get_neighbors()
            neighbors = sorted(neighbors, key=lambda obj: obj.score)
            # Generate the probability distribution of each neighbor being chosen (cannot be constant as the number of
            # neighbors can vary from one state to another)
            p = np.array([(1 / i) ** 2 for i in range(1, len(neighbors) + 1)])
            p = p / p.sum()
            self.obj = np.random.choice(neighbors, 1, p=p)[0]
            if self.obj.score < best_score:
                best_solution = self.obj
                best_score = self.obj.score
        self.obj = best_solution
        return self.obj


