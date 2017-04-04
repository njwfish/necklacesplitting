import numpy as np
import random
from src.searchable import LocalSearcheable


class Necklace(LocalSearcheable):
    """
    This class is taken from the Necklace Splitting problem, an NP-hard problem (similar to the traveling salesman), 
    which is computationally expensive to solve analytically, but can be dealt with effectively through searching 
    techniques coming out of randomized algorithms and combinatorics.
    
    The problem is, given a necklace of finite length, with a finite number of different typse of beads on it, and some 
    thieves who stole it, split the necklace in as few places as possible to generate the fairest splitting of the
    beads.
    
    Ex:
    
        beads: 0 1 1 0 0 1 1 0 1 1 0 0 1 1 0 1 1 0 0 1 1 0 1 1 0 0 1 1 0 1 1 0 0 1 1 0
        
          cuts: 0 1 1 | 0 0 1 1 0 1 1 | 0 0 1 1 | 0 1 1 | 0 | 0 1 1 0 | 1 1 0 | 0 1 1 0 1 1 0 | 0 1 1 0
        
        assign:   1           4            2         3    2      1        2           0           3
    
        thieves: 5
        
        thief beads:
        
            thief 0> 0 1 1 0 1 1 0
            
            thief 1> 0 1 1 0 1 1 0
            
            thief 2> 0 0 1 1 0 1 1 0
            
            thief 3> 0 1 1 0 1 1 0
            
            thief 4> 0 0 1 1 0 1 1
    
    This isn't a perfect split, but it is as close as possible with the number of beads and the number of thieves. 
    The problem becomes much more complex as the number of beads and thieves goes up.
    
    A necklace object primarily contains a string of integers, the beads, an integer number of thieves, who will split 
    the necklace (the list of beads), a number of cuts in the list of beads, which partition the list of beads into 
    sublists, and a list of assignments that assigns each of the partitions of beads to a specific thief.
    
    Applications of this script in bioinformatics are most obvious in detecting GC rich regions of the genome. This can
    be done by looking at the average score the script produces run for some set time on a region: this will likely be 
    able to separate these regions in a new way.
    """
    def __init__(self, beads, num_thieves, max_move=20, cuts=None, assignments=None):
        """Initialize a necklace object, randomly chooses the initial cuts and assignments if none are provided.
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
        self.cuts = list(cuts) if cuts is not None else self.gen_cuts()
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

    def __str__(self):
        """Override the default string behavior (that returns the id or the object)
        A multiline string in the following format:
        beads (the integer list of beads)
        c>cuts (the integer list of cuts),a>assignments (the integer list of assignments),s> score (the integer score)
        for each thief:
            thief (integer of the thief) > thief beads (the integer list of beads assigned to the thief)
        :return: the string
        """
        obj = str(self.beads) + '\nc>' + str(self.cuts) + ',a>' + str(self.assignments) + ',s>' + str(self.score) + '\n'
        thieves = '\n'.join([str(thief) + '>' + str(self.get_thief_beads(thief)) for thief in range(self.num_thieves)])
        return obj + thieves

    def int_beads(self, unique):
        """Takes any iterable object (e.g. a string or a list) and converts it to a list of integers
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
        """At random, generate the cuts that will partition the list of beads
        :return: sorted list of n (where n is the number of cuts) random samples from the range 0 to the number of beads
        """
        # Return the list of cuts
        return sorted(random.sample(list(range(0, self.num_beads)), self.num_cuts))

    def gen_assignments(self):
        """At random, generate the list of integers that assign each partition of beads to a thief 
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
        """Get all of the neighboring states of a necklace object. THe neighboring states are based on the new states
        that can be acchieved by a single move.
        
        There are two types of valid move for a Necklace, moving a cut (up to max_move indices) and changing an 
        assignment. A new cut cannot coincide with any preexisting cuts, and a new assignment cannot be the same as the 
        old assignment or as the assignments on either side of the partition being assigned. Allowiing multiple 
        consecutive sections to have the same assignment would be the same as removing a cut, and can make it impossible
        to find a perfect solution.
        
        :return: the list of objects for all neighboring states
        """
        neighbors = []
        # Generate all move cut moves
        for i in range(self.num_cuts):
            # Get the leftmost and rightmost positions the current cut could move to
            pos = self.cuts[i]
            right = pos - self.max_move
            right = right if right > 0 else 0
            left = pos + self.max_move
            left = left if left < self.num_beads else self.num_beads
            # For each position in the range, generate a new state with that position
            for pos in range(right, left):
                # If the position is already a cut, it cannot be the cut of this position
                if pos in self.cuts:
                    continue
                neighbors.append(Necklace(self.beads, self.num_thieves, self.max_move,
                                          cuts=np.array(self.cuts), assignments=np.array(self.assignments)))
                neighbors[-1].cuts[i] = pos
                neighbors[-1].gen_score()
        # Generate all change assignment moves
        for i in range(len(self.assignments)):
            prev = i if i > 0 else 0
            # The list of the previous, current and next assignments to ensure non of them is chosen as the new
            # assignment, which would break the number of cuts and could invalidate the algorithm
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
        """Get the beads assigned to a particular thief
        :param thief: integer thief whose beads to return
        :return: integer list of beads
        """
        thief_assignments = self.assignments == thief
        thief_beads = []
        intervals = [0] + self.cuts + [self.num_beads]
        for i in range(len(intervals)-1):
            if thief_assignments[i]:
                thief_beads += self.beads[intervals[i]:intervals[i+1]]
        return thief_beads

    def gen_score(self):
        """Generate the score, the difference in the current state and the ideal, closer to 0 is closer to the ideal  
        :return: integer score closer to 0 is closer to the ideal 
        """
        score = 0
        # Get the ideal number of beads for each thief to have (it is possible this is a fraction)
        unique, counts = np.unique(self.beads, return_counts=True)
        perfect_bead_split = counts / self.num_thieves
        # For each thief, for each bead type, count the number of beads and sum the difference of this and the ideal
        # number of beads, calculated above
        for thief in range(self.num_thieves):
            thief_beads = self.get_thief_beads(thief)
            thief_unique, thief_counts = np.unique(thief_beads, return_counts=True)
            for i in range(len(unique)):
                if i not in thief_unique:
                    score += 2
                    continue
                where = np.where(thief_unique == i)[0][0]
                score += abs(perfect_bead_split[i] - thief_counts[where])
        self.score = score
        return score
