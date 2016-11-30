import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def plot(T, filename='cluster.png'):
    tags = T['tag'].unique()
    for tag, i in enumerate(tags):
        x = T[ T['tag']==tag ]['x']
        y = T[ T['tag']==tag ]['y']
        plt.plot(x, y, 
                 marker='.',
                 linestyle=None)
        
    plt.legend(tags)
    plt.savefig(filename)
