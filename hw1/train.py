import numpy as np
import cPickle
import sys
from share import *
from feature_read import print_feature
from feature_read import featureMap

def fold4val(idxData):
    # folds[i] = mask for the training set of the i'th fold
    folds = []
    for i in range(4):
        # print 'fold', i
        mask = np.ones(idxData.shape, dtype=idxData.dtype)
        mask[:, 5*i+1 : 5*i+6 , :,:] = -1
        folds.append(mask)
    print 'Folding mask generated.'
    return folds
def validate(trainFold, w, feature, mean, std):
    # transfer the trainFold to the testFold with -1 mask
    mask = -1*np.ones(trainFold.shape, dtype=trainFold.dtype)
    testFold = dataMask(trainFold, mask)
    D = dataPreprocess(testFold, feature, withScale=False)
    D['x'] = np.array( [ (x-mean)/std for x in D['x'] ] )
    y = D['y']
    y_ = np.array( [sum( w*x ) for x in D['x']] )
    print 'y ', y
    print 'y*', y_
    foldErr = np.average(np.abs(y-y_))
    return foldErr
def dataMask(idxData,mask=0):
    if not type(mask) is np.ndarray:
        mask = np.ones(idxData.shape, dtype=idxData.dtype)
    maskData = idxData * mask
    maskData[0,:,:,:]=-1
    maskData[:,0,:,:]=-1
    return maskData
def dataPreprocess(maskData, feature, withScale=True):
# [MaskData] > Connect > Slice to consequent sequences
#            > Scan over seqs and extracet featuere to find D
# ( if withScale == True ) > feature scaling
    conData = dataConnect(maskData)
    seqData = dataSlice(conData)
    D = seqScan(seqData, feature)
    if withScale:
        D, mean, std = scale(D)
        print 'Data preprocessed with scaling.'
        return D, mean, std
    else:
        print 'Data preprocessed without scaling.'
        return D
def dataConnect(maskData):
    (M,D,I,H) = maskData.shape
    conData = np.zeros([I,M*D*H], dtype=idxData.dtype)
    s=0
    for chunk in maskData.reshape(M*D,I,H):
        conData[:,s:s+H] = chunk
        s+= H
    return conData
def dataSlice(conData):
    # seqData[seq][itemID, consequent hours] 
    seqData = []
    s=0
    for i in range( len(conData[0])-1 ):
        if conData[0,i]<0 and conData[0,i+1]>=0: #Start Point
            s = i+1
        if conData[0,i]>=0 and conData[0,i+1]<0: #End Point
            if (i+1-s) > 9:
                seqData.append(conData[:,s:i+1])
    return seqData
def seqScan(seqData, feature):
    # seqData[seq][itemID,consequent hours] 
    # feature[i] = {'src':[(itemID,hour),(itemID,hour),...]
    #               'form':function }
    # nD = |D|
    dimx = len(feature)+1

    # find nD
    nD=0
    for seq in seqData:
        nD+= len(seq[0])-9
    D = np.zeros(nD, dtype=[('x', str(dimx)+'float'),('y', 'float')])
    
    # extract feature
    p=0
    for seq in seqData:
        for i in range(len(seq[0])-9):
            raw = seq[:,i:i+10]
            # raw[item][hour]
            D[p+i] = featureExt(raw, feature)
        p+= i+1
    return D
def scale(D):
    dimx = len( D[0]['x'] )
    mean = np.mean(D['x'], axis=0)
    std  = np.std (D['x'], axis=0)
    mean[0], std[0] = 0, 1
    D['x'] = np.array( [ (x-mean)/std for x in D['x'] ] )
    return D, mean, std
'''
idxData=np.random.randint(10,size=(3,3,2,24))+1
print idxData
mask=np.ones(idxData.shape, dtype=idxData.dtype)
print mask
data=dataSlicer(idxData,mask)
print data
feature=[{'src':[(0,4),(0,7)],
          'form':sum},
         {'src':[(0,6),(0,5)],
          'form':sum},
         {'src':[(0,3)],
          'form':sum}]

print findTrainSet(data,feature)
'''
def GradientLoss(D, w, rate):
    gradient = np.zeros(len(w), dtype='float')
    for i in range( len(w) ):
        for (x,y) in D:
            err = y-sum(w*x)
            gradient[i] += err*x[i]
        gradient[i] *= -2
        gradient[i] += rate*2*w[i]
    return gradient
def meanErr(D,w):
    err = 0
    for (x,y) in D:
        err += np.abs( y-sum(w*x) )
    return err/len(D)
def Loss(D, w):
    l = 0
    for (x,y) in D:
        err = y-sum(w*x)
        l += err*err
    l /= len(D)
    l = pow(l, 0.5)
    return l
def reg(w):
    reg = 0
    for x in w:
        reg += x*x
    return reg

def LA_Regression(D):
    print 'Training...'
    dimx = len(D[0]['x'])
    A = np.matrix(D['x'])
    b = np.matrix(D['y']).T
    w = np.matrix(np.zeros(dimx, dtype='float'))
    w = (A.T*A).I*A.T*b
    print 'Complete training.\n'
    print 'w', w.T
    return np.array(w.T)[0]
    
def GD_Regression(D, regRate, GDpara):
    # model y= sum_item(w[item]x_item)
    print 'Training...'
    dimx = len(D[0]['x'])
    eta, itr, init = GDpara.eta, GDpara.itr, GDpara.init

    w = np.zeros( dimx, dtype='float' )
    G = np.zeros( dimx, dtype='float' )
    X = D['x']
    y = D['y']
    w[ itemIdx['PM2.5'] ] = init #initial value
    if itr>0:
        for i in range(int(itr)):
            g = -2* np.dot(( y-np.dot(X,w)), X) + 2*regRate*w
            G += g*g
            w = w-eta*g/np.power(G,0.5)
            print i, meanErr(D,w), reg(w)
    else:
        stopRate = -1*itr
        w_last = np.zeros(dimx, dtype='float')
        changeRate = stopRate+1
        while changeRate >= stopRate:
            g = -2* np.dot(( y-np.dot(X,w)), X) + 2*regRate*w
            G += g*g
            change = eta*g/np.power(G,0.5)
            w = w-change
            changeRate = np.linalg.norm(change)/np.linalg.norm(w)
            print changeRate, meanErr(D,w), reg(w)
    print 'Complete training.\n'
    return w

def trainD(D, feature, trainConf):
    if trainConf.method == 'la':
        w = LA_Regression(D)
    if trainConf.method == 'gd':
        w = GD_Regression(D, trainConf.regRate, trainConf.GDpara)
        if trainConf.regRate == 0:
            LA_Regression(D)
    return w

def trainWithValidation(idxData, feature, trainConf):
    print 'Validation:'
    trainFolds = []
    avgValErr = 0

    for i,mask in enumerate(fold4val(idxData)):
        print '\n\nFold', i
        maskData = dataMask(idxData, mask)
        D, mean, std = dataPreprocess(maskData, feature)
        w = trainD(D, feature, trainConf)               #train

        print 'Validate fold', i
        foldErr = validate(maskData, w, feature, mean, std)
        avgValErr += foldErr
        print 'foldErr', foldErr

    avgValErr /= 4
    print '\n\n\n####################\n4-fold validation Done.'
    print 'AvgValErr:', avgValErr
    print 'Feature terms:', len(feature)
    print '####################\n\n\n'
    return avgValErr

def trainAll(idxData, feature, para):
    print '\n\nAll Data'
    if para.flt:
        D, mean, std, maskData, w = [], [], [], [], []
        for mask in para.mask:
            maskData_ = dataMask(idxData, mask)
            D_, mean, std = dataPreprocess(maskData_, feature)
            w_ = trainD(D_, feature, para)

            D.append(D_)
            mean.append(mean_)
            std.append(std_)
            w.append(w_)
        w = np.array(w)
    else:
        D, mean, std = dataPreprocess(idxData, feature)
        w = trainD(D, feature, para)
    return w, mean, std

def scale(D):
    dimx = len( D[0]['x'] )
    mean = np.mean(D['x'], axis=0)
    std  = np.std (D['x'], axis=0)
    mean[0], std[0] = 0, 1
    D['x'] = np.array( [ (x-mean)/std for x in D['x'] ] )
    return D, mean, std

if __name__ == '__main__':
### initialize
    filename = 'trainingData.m'
    feature = []
    val = True
    print 'Feature generating...'
    for h in range(9):
        for i in range(17):
            feature.append({'src':[(i,h)], 'form':0})
    # feature[i] = {'src':[(itemID,hour),(itemID,hour),...]
    #               'form':function }
    print_feature( featureMap(feature) )
    print 'Feature generated with dim', len(feature)
    para = config( len(feature)+1 )

### Read original idxData
    with open(filename, 'r') as f:
        idxData=cPickle.load(f)

### Change Configuration
    wfname = sys.argv[0][:-9]
    for i in range( 1, len(sys.argv) ):
        arg = sys.argv[i]
        wfname += arg
        # Gradient Descent Parameters
        if arg.startswith('-itr'):
            para.GDpara.itr = float(arg[4:])
        if arg.startswith('-eta'):
            para.GDpara.eta[:] = float(arg[4:])
        if arg.startswith('-init'):
            para.GDpara.init = float(arg[5:])

        # Model
        if arg.startswith('-reg'):
            if arg == '-reg':
                # Autotuning regRate
                para.regRate = -1
            para.regRate = float(arg[4:])
        if arg == '-flt':
            val = False
            with open('mostSim.m', 'r') as f:
                mostSim = cPickle.load(f)
            mask = []
            for i, prob in enumerate(mostSim):
                print 'filter for prob', i, prob
                month, hour = prob[0], prob[1]
                mask_ = -1*np.ones(idxData.shape, dtype=idxData.dtype)
                if hour < 14:
                    mask_[month,   11:, :, hour:hour+10]=1
                    mask_[month+1, :11, :, hour:hour+10]=1
                else:
                    mask_[month,   11:, :, hour:] = 1
                    mask_[month,   11:, :, :hour-14] = 1
                    mask_[month+1, :11, :, hour:] = 1
                    mask_[month+1, :11, :, :hour-14] = 1
                    print 'Across the day'
                mask.append(mask_)
            para.setFlt(mask)
        if arg.startswith('-feat'):
            with open(sys.argv[i+1], 'r') as f:
                feature = cPickle.load(f)
            if sys.argv[i+1][-2:] == '.w':
                feature = feature['feature']
            print '\nFeature changed to', sys.argv[i+1], 'with dim', len(feature)
            print_feature( featureMap(feature) )
            para.setDim( len(feature)+1 )

        # Training Method
        if arg == '-la':
            para.method = 'la'
        if arg == '-NOval':
            val = False
    if para.regRate == 0:
        para.method = 'la'

### Train with validation
    if val == True:
        avgValErr = trainWithValidation(idxData, feature, para)
        '''
        x = raw_input('Continue(q for quit)?')
        if x == 'q':
            sys.exit()
        '''
        wfname = 'val%06.2f' % avgValErr + wfname

### Train with all data
    w, mean, std = trainAll(idxData, feature, para)

### Save the result
    wfname += '.w'
    save = raw_input('Save '+wfname+' ?')
    name = raw_input('File Name:')
    if save == '1':
        if name != '':
            wfname = name
        with open(wfname, 'wb') as f:
            cPickle.dump( {'w':w, 
                'feature':feature, 
                'mean':mean, 
                'std':std},         f)
            print wfname+' saved!'
    else:
        print 'Pass.'
