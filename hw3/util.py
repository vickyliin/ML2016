import numpy as np
import cPickle

def load_data(path, data=['u', 't']):
    print 'Loading data...'
    all_unlabel, all_test = 0, 0
    all_label = cPickle.load( open('%s/all_label.p' % path, 'rb') )
    all_label = np.reshape(all_label, (-1, 3, 32, 32))
    if 'u' in data:
        all_unlabel = cPickle.load( open('%s/all_unlabel.p' % path, 'rb') )
        all_unlabel = np.reshape(all_unlabel, (-1, 3, 32, 32))
    if 't' in data:
        all_test = cPickle.load( open('%s/test.p' % path, 'rb') )
        all_test = np.reshape(all_test['data'], (-1, 3, 32, 32))


    label_mean = np.mean(all_label)

    # normalization
    label_mean = np.reshape(np.mean(all_label, axis=0), (3, 32, 32))
    all_label = (all_label - label_mean) / 255
    if 'u' in data:
        all_unlabel = (all_unlabel - label_mean) / 255
    if 't' in data:
        all_test = (all_test - label_mean) / 255

    print 'Loaded.'

    return all_label, all_unlabel, all_test
