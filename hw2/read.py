import numpy as np
import cPickle
import sys
from pprint import pprint

if __name__ == '__main__':
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        m = cPickle.load(f)
    pprint(m)
