import cPickle
import sys
import numpy as np

trainingDataName = 'data/train.csv'
trainingDataAmt = 97920
testDataName = 'data/test_X.csv'
probAmt = 240
items = ['PM2.5', 'AMB_TEMP','CH4','CO','NMHC',
        'NO','NO2', 'NOx','O3','PM10','RH','SO2','THC',
        'WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
itemIdx = {}
for i,item in enumerate(items):
    itemIdx[item]=i
print itemIdx
if __name__ == '__main__':
    for arg in sys.argv[1:]:
        if arg == '-train':
            # idxData[month, day, item, hour]
            idxData = np.ones([13,21,len(items),24], dtype='float')
            with open(trainingDataName, 'r') as f:
            # 2014/1/1,Station,PM10,56,50,48,35,25,12,4,2,11,38,56,64,56,57,52,51,66,85,85,63,46,36,42,42
                for line in f:
                    raw = line.strip().split(',')
                    raw[0] = raw[0].split('/')
                    month, day, itemID = int(raw[0][1]), int(raw[0][2]), itemIdx[raw[2]]
                    value = np.array(raw[3:]).astype('float')
                    idxData[month,day,itemID] = value
            print idxData[:,:,0,:]
            #s = raw_input('Save -train?')
            #if s == '1':
            with open('trainingData.m', 'wb') as f:
                cPickle.dump(idxData, f)
                print 'trainingData.m Saved!'

        if arg == '-test':
            # data[probID, item, hour]
            data = np.zeros([probAmt, len(items), 9], dtype='float')
            with open(testDataName, 'r') as f:
            # id_0,PM2.5,27,13,24,29,41,30,29,27,28
                for line in f:
                    raw = line.strip().split(',')
                    if raw[1] == 'RAINFALL':
                        continue
                    probID, itemID = int(raw[0][3:]), itemIdx[raw[1]]
                    value = np.array(raw[2:]).astype('float')
                    data[probID,itemID] = value
            print data[10]
            #s = raw_input('Save -test?')
            #if s == '1':
            with open('testData.m', 'wb') as f:
                cPickle.dump(data, f)
                print 'testData.m Saved!'
