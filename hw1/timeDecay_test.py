import cPickle
import numpy as np
import sys

testDataAmt = 43200
testProbAmt = 240
itemIdx = {}

def preprocess(fname, decayRate, feature):
    dimx = len(feature)+1
    i = 1
    for item in feature:
        itemIdx[item] = i
        i += 1
    print itemIdx
 
    decay = np.exp( np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,0])*decayRate )
    x = np.zeros( [testProbAmt, dimx] , dtype='float' )
    with open(fname) as f:
        testData = cPickle.load(f)
    x[:,0] = np.ones(testProbAmt, dtype='float')
    for d in testData:
        if d['item'] in feature:
            i = int(d['ProbID'][3:])
            x[i][ itemIdx[ d['item'] ] ] += decay[ d['hour'] ] * d['value']
    return x

if __name__ == '__main__':
    decayRate = 1
    wfname = sys.argv[0][:-8]
    feature = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2',
            'NOx','O3','PM10','PM2.5','RH','SO2','THC',
            'WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
    for i in range( 1, len(sys.argv) ):
        arg = sys.argv[i]
        wfname += arg
        if arg.startswith('-itr'):
            itr = float(arg[4:])
        if arg.startswith('-eta'):
            eta = np.array( [ float(arg[4:]) ]*2 )
        if arg.startswith('-decay'):
            decayRate = float(arg[6:])
        if arg.startswith('-reg'):
            regRate = float(arg[4:])
        if arg.startswith('-init'):
            init = float(arg[5:])
        if arg.startswith('-open'):
            feature = arg[5:].split(',')
        if arg.startswith('-close'):
            for item in arg[6:].split(','):
                feature.remove(item)
    with open(wfname+'.w', 'r') as f:
        w = cPickle.load(f)

    x = preprocess('testData.m', decayRate, feature)
    y = [ sum(w*x[i]) for i in range(len(x))]
    print 'id,value'
    for i in range(len(y)):
        print 'id_'+str(i)+','+str(y[i])
    save = raw_input('Save?')
    if save != '1':
        sys.exit()
    with open(wfname+'.csv', 'w') as f:
        f.write('id,value\n')
        for i in range(len(y)):
            f.write('id_'+str(i)+','+str(y[i])+'\n')
        print wfname+'.csv saved!'
