from data import *
from train import *
import sys
path = sys.argv[1]
output = sys.argv[2]

C = loadC(path)     # load questions
tags = loadtags()   # load model
ans = check(C,tags) # check answer
save(ans,output)    # save answer

