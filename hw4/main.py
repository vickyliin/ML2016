from data import *
from train import *
import sys
path = sys.argv[1]
output = sys.argv[2]

C = loadC(path)         # load questions
T = loadT()             # load model
ans = check(C,T['tag']) # check answer
save(ans,output)        # save answer
