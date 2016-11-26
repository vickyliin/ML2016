from data import *
from train import *
import sys
path = sys.argv[1]
output = sys.argv[2]

print('Load check_index.csv')
C = loadC(path)
print('Load model')
table = loadtable()
print('\nCheck answer')
ans = check(C, table)
print('Save')
save(ans, output)
