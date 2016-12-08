import matplotlib
matplotlib.use('Agg')
import plotly.plotly as py
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd

def decompose(M, dimOut=2, funct=TruncatedSVD, **kwargs):
    model = funct(n_components=dimOut, **kwargs)
    M = model.fit_transform(M)
    x,y = tuple(zip(*M))
    x,y = np.array(x), np.array(y)
    return x,y

def readL(filename='label_StackOverflow.txt'):
    with open(filename) as f:
        tags = [int(line.strip()) for line in f]
    return tags

tags = ['autoencoder', 'lsa-40', 'lsa-20', 'label']

def keytermExt(T, F, terms, nb_keyterm=5, tags=tags):
    keyterm_dict = {} 
    n = nb_keyterm
    terms = np.array(terms)
    for tag in tags:
        keyterm_tag = {}
        for i,cls in enumerate(T[tag].unique()):
            Ftag = F[ T[ T[tag]==cls ].index ].toarray()
            keyterm_tag[cls] = terms[ sum(Ftag).argsort()[::-1][:n] ]
        keyterm_dict[tag] = pd.DataFrame.from_dict(keyterm_tag, orient='index')
    return keyterm_dict

def plot(T, tags=tags, n=2, s=0.1):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    x, y = T['x'], T['y']
    stdx, stdy = np.std(x), np.std(y)
    meanx, meany = np.mean(x), np.mean(y)
    xl, xr = meanx-n*stdx, meanx+n*stdx
    yl, yr = meany-n*stdy, meany+n*stdy
    xlim = (min(x[x>xl]), max(x[x<xr]))
    ylim = (min(y[y>yl]), max(y[y<yr]))

    plt.clf()
    if type(tags)!=list:
        plt.figure(figsize=(10,10))
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        tag='unlabeled'
        plt.scatter(T['x'], T['y'],
                    s=s, )
        plt.title(tag)
        plt.axis('off')
        plt.savefig(tag+'.png', bbox_inches="tight")
        print('%s.png saved!' % tag)
        plt.close()
        return 0

    for tag in tags:
        plt.figure(figsize=(10,10))
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        color = cm.rainbow(np.linspace(0, 1, len(T[tag].unique())))
        for i,c in enumerate(T[tag].unique()):
            x = T[ T[tag] == c ]['x']
            y = T[ T[tag] == c ]['y']
            plt.scatter(x,y, 
                    color=color[i],
                    s=s,)
            plt.title(tag)
            plt.axis('off')
        plt.savefig(tag+'.png', bbox_inches="tight")
        print('%s.png saved!' % tag)
        plt.close()
    return 1
 
