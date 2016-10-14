import numpy as np

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
        self.method = 'gd'
        self.flt = False
        self.mask = 0
    def setDim(self, dim):
        self.dim = dim
        self.GDpara.setDim(dim)
    def setFlt(self, mask):
        self.flt = True
        self.mask = mask
class eqn:
    def __init__(self, kind, coef, weight=1):
        self.kind = kind
        self.coef = coef
        self.weight = weight
    def call(self, x):
        if self.kind.startswith('pow'):
            y = np.power(x, self.coef)
            return sum( y*self.weight )
        if self.kind.startswith('exp'):
            y = np.exp(self.coef*x)
            return sum( y*self.weight )
        if self.kind.startswith('mul') or self.kind.startswith('pi'):
            s = 1
            x = np.power(x, self.coef)
            for xi in x:
                s *= xi
            return s

def featureExt(raw, feature):
    dimx = len(feature)+1
    d = np.zeros(1, dtype=[('x', str(dimx)+'float'),('y', 'float')])[0]
    # d: an element in D
    for j,term in enumerate(feature):
        rawExt=[]
        for src in term['src']:
            rawExt.append(raw[src])
        rawExt = np.array(rawExt)
        if term['form'] == 0:
            d['x'][j+1] = rawExt[0]
        else:
            d['x'][j+1] = term['form'].call(rawExt)
    try:
        # for testing data process, raw.shape=[17,9]
        d['y'] = raw[0,9]
    except:
        pass
    d['x'][0] = 1
    return d

def print_feature(fmap):
    print('\t'),
    for i in range(9):
        print(str(i)+'\t'),
    print 
    for row,item in zip(fmap,items):
        print(item+'\t'),
        for x in row:
            print(x+'\t'),
        print
def featureMap(feature):
    fmap = np.zeros([17,9], dtype='S50')
    for i,term in enumerate(feature):
        form = term['form']
        for (itemID,hour) in term['src']:
            if form == 0:
                fmap[itemID, hour] += 'x,'
            else:
                fmap[itemID, hour] += form.kind[:1]+str(form.coef)+','
    return fmap
