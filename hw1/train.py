import numpy as np
import cPickle
import sys

items = ['PM2.5', 'AMB_TEMP','CH4','CO','NMHC',
'NO','NO2', 'NOx','O3','PM10','RH','SO2','THC',
'WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
itemIdx = {}
for i,item in enumerate(items):
    itemIdx[item]=i
print itemIdx
# idxData[month, day, itemID, hour]

class GDconf:
    def __init__(self, dim):
        self.dim = dim
        self.itr = 100
        self.eta = np.array([1]*dim)
        self.init = 1 # initial PM2.5 weight
    def setDim(self, dim):
        self.dim = dim
        self.eta = np.array([self.eta[0]]*dim)
class config:
    def __init__(self, dim):
        self.dim = dim
        self.GDpara = GDconf(dim)
        self.regRate = 0
        self.la = False
        self.flt = False
    def setDim(self, dim):
        self.dim = dim
        self.GDpara.setDim(dim)
        
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

def validate(idxData, w):
    # foldErr[i] = the mean error as i'th fold validation
    # w[i] = the training result w of the i'th fold validation
    print '\nValidating the results...'
    foldErr = [0]
    masks = [ m*-1 for m in fold4val(idxData) ]
    seqValData = [ dataSlicer(idxData,m) for m in masks ]
    valSets = [ seqToD(x,feature) for x in seqValData ] 
    for i,D in enumerate(valSets):
        print 'fold', i
        x,y = D['x'], D['y']
        y_ = np.array([ sum(w[i]*xx) for xx in x ])
        avgErr = np.average(np.abs(y-y_))
        foldErr.append(avgErr)
        print 'y:', y[:5]
        print 'y*', y_[:5]
        print 'AvgErr:', avgErr
        print ''
    return foldErr

def findTrainSet(idxData, feature, val=False):
    # trainSets: 4 items D, different fold
    # trainSet: D (all)
    if val:
        print '\nSplitting the training set...'
        masks = fold4val(idxData)
        seqData = [ dataSlicer(idxData,m) for m in masks ]
        trainSets = [ seqToD(x,feature) for x in seqData ] 
        for i in range(4):
            print 'Fold', i, trainSets[i][0], '...'
        print 'Training set split.'
        print ''
        return trainSets
    else:
        seqData = dataSlicer(idxData)
        trainSet = seqToD(seqData, feature)
        return trainSet
'''
def findMask(idxData, area):
    # area[i] = [[startMonth,endMonth],[startDay,endDay],[startHour,endHour]]
    # mask[...]={-1:mask,1:remain}
    mask = -1*np.ones(idxData.shape, dtype=idxData.dtype) 
    for a in area:
        mask[ a[0,0]:a[0,1]+1, a[1,0]:a[1,1]+1, :, a[2,0]:a[2,1]+1 ] = 1
    return mask
'''
def dataSlicer(idxData, mask=0):
    # idxData[month, day, itemID, hour]
    # seqData[seq][itemID, consequent hours] 
    if not type(mask) is np.ndarray:
        print 'Slicing for all train set...'
        mask = np.ones(idxData.shape, dtype=idxData.dtype)
    seqData = []
    mask[0,:,:,:]=-1
    mask[:,0,:,:]=-1
    (M,D,I,H) = idxData.shape
    conData = np.zeros([I,M*D*H], dtype=idxData.dtype)
    s=0
    for chunk in (idxData*mask).reshape(M*D,I,H):
        conData[:,s:s+H] = chunk
        s+= H
    s=0
    for i in range( len(conData[0])-1 ):
        if conData[0,i]<0 and conData[0,i+1]>=0: #Start Point
            s = i+1
        if conData[0,i]>=0 and conData[0,i+1]<0: #End Point
            if (i+1-s) > 9:
                seqData.append(conData[:,s:i+1])
    return seqData
def featureApply(raw, feature):
    dimx = len(feature)+1
    d = np.zeros(1, dtype=[('x', str(dimx)+'float'),('y', 'float')])
    # d: an element in D
    for j,term in enumerate(feature):
        rawExt=[]
        for src in term['src']:
            rawExt.append(raw[src])
        #print 'j:', j, ',rawExt:', rawExt
        if term['form'] == 0:
            d[0]['x'][j+1] = rawExt[0]
        else:
            d[0]['x'][j+1] = term['form'](rawExt)
    try:
        # for testing data process, raw.shape=[17,8]
        d[0]['y'] = raw[0,9]
    except:
        pass
    d[0]['x'][0] = 1
    return d[0]
def seqToD(seqData, feature):
    # seqData[seq][itemID,consequent hours] 
    # feature[i] = {'src':[(itemID,hour),(itemID,hour),...]
    #               'form':function }
    # nD = |D|
    dimx = len(feature)+1
    nD=0
    for seq in seqData:
        nD+= len(seq[0])-9
    D = np.zeros(nD, dtype=[('x', str(dimx)+'float'),('y', 'float')])
    p=0
    for seq in seqData:
        for i in range(len(seq[0])-9):
            raw = seq[:,i:i+10]
            # raw[item][hour]
            D[p+i] = featureApply(raw, feature)
        p+= i+1
    return D
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
    print '\nTraining...'
    dimx = len(D[0]['x'])
    A = np.matrix(D['x'])
    b = np.matrix(D['y']).T
    w = np.matrix(np.zeros(dimx, dtype='float'))
    w = (A.T*A).I*A.T*b
    print w.T
    print 'Complete training.'
    return np.array(w.T)[0]
    
def GD_Regression(D, regRate, GDpara):
    # model y= sum_item(w[item]x_item)
    print '\nTraining...'
    dimx = len(D[0]['x'])
    eta, itr, init = GDpara.eta, GDpara.itr, GDpara.init

    w = np.zeros( dimx, dtype='float' )
    G = np.zeros( dimx, dtype='float' )
    w[ itemIdx['PM2.5'] ] = init #initial value
    if itr>0:
        for i in range(itr):
            g = GradientLoss(D, w, regRate)
            G += g*g
            w = w-eta*g/np.power(G,0.5)
            print i, w, Loss(D,w), reg(w)
    else:
        stopRate = -1*itr
        w_last = np.zeros(dimx, dtype='float')
        changeRate = np.ones(dimx, dtype='float')
        while abs(np.average(changeRate)) >= stopRate:
            g = GradientLoss(D, w, regRate)
            print g
            G += g*g
            changeRate = eta*g/np.power(G,0.5)
            w = w-changeRate
            changeRate = changeRate/w
            print np.average(changeRate), w, Loss(D,w), reg(w)
    print 'Complete training.'
    return w

def trainWithValidation(idxData, feature, trainConf):
    folds = findTrainSet(idxData, feature, val=True)
    if trainConf.la:
        w = [ LA_Regression(D) for D in folds ]
    else:
        w = []
        for i,D in enumerate(folds):
            print '\nFold', i
            w.append(  GD_Regression(D, trainConf.regRate, trainConf.GDpara) )
            if trainConf.regRate == 0:
                print LA_Regression(D)
    foldErr = validate(idxData, w)
    avgValErr = sum(foldErr)/4
    print 'foldErr:', foldErr, 'Avg:', avgValErr
    return avgValErr
def trainAll(idxData, feature, para):
    D = findTrainSet(idxData, feature)
    if para.la:
        w = LA_Regression(D)
    else:
        w = GD_Regression(D, para.regRate, para.GDpara)
        if regRate == 0:
            print LA_Regression(D)
    return w

if __name__ == '__main__':
### initialize
    filename = 'trainingData.m'
    feature = [{'src':[(0,8)], 'form':0},
            {'src':[(0,7)], 'form':0},
            {'src':[(0,6)], 'form':0},
            {'src':[(0,5)], 'form':0},
            {'src':[(0,4)], 'form':0}]
    # feature[i] = {'src':[(itemID,hour),(itemID,hour),...]
    #               'form':function }
    para = config( len(feature)+1 )

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
            para.flt = True

        # Training Method
        if arg == '-la':
            para.la = True
    print 'Feature:', feature

### Read original idxData
    with open(filename, 'r') as f:
        idxData=cPickle.load(f)

### Train with validation
    avgValErr = trainWithValidation(idxData, feature, para)
    x = raw_input('Continue(q for quit)?')
    if x == 'q':
        sys.exit()

### Train with all data
    w = trainAll(idxData, feature, para)

### Save the result
    wfname = 'val%06.2f' % avgValErr + wfname
    wfname += '.w'
    save = raw_input('Save '+wfname+' ?')
    if save == '1':
        with open(wfname, 'wb') as f:
            cPickle.dump({'w':w, 'feature':feature},f)
            print wfname+' saved!'
    else:
        print 'Pass.'
