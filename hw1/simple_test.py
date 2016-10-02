import cPickle
import numpy as np

testDataAmt = 43200

def preprocess(fname):
    x = []
    with open(fname) as f:
        testData = cPickle.load(f)
    for d in testData:
        if d['item'] == 'PM2.5' and d['hour'] == 8:
            x.append(d['value'])
    return np.array(x)

if __name__ == '__main__':
    with open('trained_w.m', 'r') as f:
        w = cPickle.load(f)
    x = preprocess('testData.m')
    y = float(w[0]) + float(w[1])*x
    
    print 'id,value'
    for i in range(len(y)):
        print 'id_'+str(i)+','+str(int(y[i]))
