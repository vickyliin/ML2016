import cPickle
import sys
import numpy as np

trainingDataName = 'data/train.csv'
trainingDataAmt = 97920
testDataName = 'data/test_X.csv'
testDataAmt = 43200

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        data = np.zeros(trainingDataAmt, dtype=[
            ('month', 'u1'),
            ('day', 'u1'),
            ('hour', 'u1'),
            ('item', 'S10'),
            ('value', 'float')])
            
                    
        with open(trainingDataName, 'r') as f:
            i = 0
            for line in f:
                raw = line.strip().split(',')
                raw[0] = [int(x) for x in raw[0].split('/')]
                hour = 0
                print raw 
                for value in raw[3:]:
                    data[i] = (raw[0][1], raw[0][2], hour, raw[2], float(value)) 
                    hour += 1
                    print i, data[i]
                    i += 1
        with open('trainingData.m', 'wb') as f:
            cPickle.dump(data, f)

    if sys.argv[1] == 'test':
        data = np.zeros(testDataAmt, dtype=[
            ('ProbID', 'S10'),
            ('hour', 'u1'),
            ('item', 'S10'),
            ('value', 'float')])
        with open(testDataName, 'r') as f:
            i = 0
            for line in f:
                raw = line.strip().split(',')
                hour = 0
                for value in raw[2:]:
                    if value == 'NR':
                        value = '-1'
                    data[i] = (raw[0],  hour, raw[1], float(value))
                    hour += 1
                    print i, data[i]
                    i += 1
        with open('testData.m', 'wb') as f:
            cPickle.dump(data, f)
