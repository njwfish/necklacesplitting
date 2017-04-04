import argparse
from src.necklace import Necklace
from src.localsearch import LocalSearch

# Setup the parser to take command line arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--in', help='The input file containing a string of "beads"', required=True)
parser.add_argument('-o', '--out', help='The file to write out to', required=True)
parser.add_argument('--n', default=2, help='The number of "thieves" to split the beads for')
parser.add_argument('--t', default=2, help='The amount of time to run the search for')
args = vars(parser.parse_args())

# Read in the file
with open(args['in']) as file_in:
    beads = file_in.read()

# Create the Necklace object
n = Necklace(beads, int(args['n']))

# Perform the local search for the number of seconds passed in args
a = LocalSearch(n).search(int(args['t']))

with open(args['out'],  'a') as file_out:
    # Write out the necklace object found by the search
    file_out.write(str(a))
