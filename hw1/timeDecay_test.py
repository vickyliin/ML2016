import cPickle
import numpy as np
import sys

testDataAmt = 43200
testProbAmt = 240
with open(sys.argv[0][:-8]+'-itemIdx', 'r') as f:
    itemIdx = cPickle.load(f)

def preprocess(fname, decayRate):
    decay = np.exp( np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,0])*decayRate )
    x = np.zeros( [testProbAmt, len(itemIdx)+1] , dtype='float' )
    with open(fname) as f:
        testData = cPickle.load(f)
    for xx in x:
        xx[0] = 1
    for d in testData:
        if d['ProbID'] != '' and d['item'] != 'RAINFALL':
            i = int(d['ProbID'][3:])
            x[i][ itemIdx[ d['item'] ] ] += decay[ d['hour'] ] * d['value']
    return x

if __name__ == '__main__':
    decayRate = 1
    wfname = sys.argv[0][:-8]
    for i in range( 1, len(sys.argv) ):
        arg = sys.argv[i]
        wfname += arg
        if arg.startswith('-itr'):
            itr = int(arg[4:])
        if arg.startswith('-eta'):
            eta = np.array( [ float(arg[4:]) ]*2 )
        if arg.startswith('-decay'):
            decayRate = float(arg[6:])
        if arg.startswith('-reg'):
            regRate = float(arg[4:])
    wfname += '.w'
    with open(wfname, 'r') as f:
        w = cPickle.load(f)
#        print 'File %s is loaded!' % wfname

    x = preprocess('testData.m', decayRate)
    y = [ sum(w*x[i]) for i in range(len(x))]
    print 'id,value'
    for i in range(len(y)):
        print 'id_'+str(i)+','+str(y[i])
