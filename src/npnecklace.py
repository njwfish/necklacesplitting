import numpy as np
import random
from src.searchable import LocalSearcheable


class Necklace(LocalSearcheable):
    def __init__(self, beads, num_thieves, max_move=20, cuts=None, assignments=None):
        """Initialize a necklace object
        :param beads: the iterable beads, convertible to a list of integers
        :param num_thieves: the number of sublists to create
        :param max_move: the maximum distance to move a cut
        :param cuts: a list of preassigned cuts (for copying or starting searching from a specific point)
        :param assignments: a list of preassigned assignments (for copying or starting searching from a specific point)
        """
        self.beads = list(beads)
        self.num_beads = len(beads)
        unique = np.unique(self.beads)
        self.bead_types = len(unique)
        self.int_beads(unique)
        self.num_thieves = num_thieves
        self.max_move = max_move
        self.num_cuts = (num_thieves - 1) * self.bead_types
        self.cuts = np.array(cuts) if cuts is not None else self.gen_cuts()
        self.assignments = assignments if assignments is not None else self.gen_assignments()
        self.score = 0
        self.gen_score()

    def __eq__(self, other):
        """Override the default equality test
        :param other: any other object
        :return: boolean whether the two objects are equal based on __dict__ comparisons
        """
        if isinstance(other, self.__class__):
            for k in self.__dict__:
                if not np.array_equal(self.__dict__[k], other.__dict__[k]):
                    return False
            return True
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test
        :param other: any other object
        :return: boolean on whether the two objects are equal
        """
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)
        :return: hash of the sorted items in the dict of this object, used as an identifier instead of object id
        """
        return hash(tuple(sorted(self.__dict__.items())))

    def int_beads(self, unique):
        """
        Takes any iterable object (e.g. a string or a list) and converts it to a list of integers
        :param unique: the unique items in the string
        """
        # If the beads is a list of ints, return
        if all(isinstance(item, int) for item in self.beads):
            return
        # Create a dict mapping the objects in the list to integers
        to_int = {unique[i]: i for i in range(self.bead_types)}
        # Remap the list of objects to integers
        self.beads = [to_int[bead] for bead in self.beads]

    def gen_cuts(self):
        """
        At random, generate the cuts that will partition the list of beads
        :return: sorted list of n (where n is the number of cuts) random samples from the range 0 to the number of beads
        """
        # Return the list of cuts
        return np.sort(np.random.choice(self.num_beads, self.num_cuts))

    def gen_assignments(self):
        """
        At random, generate the list of integers that assign each partition of beads to a thief 
        :return: list of integers assigning each partition of beads to a thief
        """
        # Get a list of zeros for each partition in the beads
        assignments = np.zeros(self.num_cuts + 1)
        # The list of all possible assignments, equivalent to the list of thieves
        all_assignments = np.arange(0, self.num_thieves)
        # The list of possible assignments for the next position (cannot be the same as the last assignment)
        possible_assignments = np.arange(1, self.num_thieves)
        for i in range(1, self.num_cuts + 1):
            # Select a random thief to assign this partition to
            assignments[i] = random.choice(possible_assignments)
            possible_assignments = [a for a in all_assignments if assignments[i] != a]
        # Return the list of assignments
        return assignments

    def get_neighbors(self):
        neighbors = []
        for i in range(self.num_cuts):
            pos = self.cuts[i]
            right = pos - self.max_move
            right = right if right > 0 else 0
            left = pos + self.max_move
            left = left if left < self.num_beads else self.num_beads
            for pos in range(right, left):
                if pos in self.cuts:
                    continue
                neighbors.append(Necklace(self.beads, self.num_thieves, self.max_move,
                                          cuts=np.array(self.cuts), assignments=np.array(self.assignments)))
                neighbors[-1].cuts[i] = pos
                neighbors[-1].gen_score()
        for i in range(len(self.assignments)):
            prev = i if i > 0 else 0
            cannot_assign = self.assignments[prev:prev + 3]
            for thief in range(self.num_thieves):
                if thief in cannot_assign:
                    continue
                neighbors.append(Necklace(self.beads, self.num_thieves, self.max_move,
                                          cuts=np.array(self.cuts), assignments=np.array(self.assignments)))
                neighbors[-1].assignments[i] = thief
                neighbors[-1].gen_score()
        return neighbors

    def get_thief_beads(self, thief):
        intervals = np.insert(np.array([0, self.num_beads]), 1, self.cuts)
        intervals = np.stack((intervals[:-1], intervals[1:]), axis=1)
        return np.concatenate([self.beads[start:end] for start, end in intervals[self.assignments == thief]])

    def gen_score(self):
        """
        Generate the score, the difference in the current state and the ideal, closer to 0 is closer to the ideal  
        :return: integer score closer to 0 is closer to the ideal 
        """
        score = 0
        # Get the ideal number of beads for each thief to have (it is possible this is a fraction)
        unique, counts = np.unique(self.beads, return_counts=True)
        perfect_bead_split = counts / self.num_thieves
        thief_beads = [np.array(self.get_thief_beads(thief)) for thief in range(self.num_thieves)]
        bead_types = np.arange(0, self.bead_types)
        thief_counts = np.array([np.sum(np.equal(np.tile(beads, (self.bead_types, 1)),
                                        np.tile(bead_types, (len(beads), 1)).T), axis=1)
                                for beads in thief_beads])
        self.score = np.sum(np.abs(thief_counts-perfect_bead_split))
        return score
