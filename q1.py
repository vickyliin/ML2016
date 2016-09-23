import sys
import numpy as np

if __name__ == '__main__':
    colNo = sys.argv[1]
    data = sys.argv[2]

    a = np.zeros( [0,11] , dtype='float')
    with open(data, 'r') as f:
        for line in f:
            row = np.expand_dims(np.array(line.split()), axis=0)
#            print row, a
            a = np.concatenate((a,row), axis=0)
            a = a.astype('float')
#    print a


    with open('ans1.txt', 'w') as f:
        ans = np.sort(a[:, colNo])
        ans_str = ''
        first = True
        for x in ans:
            if first:
                first = False
            else:
                ans_str += ','
            ans_str += str(x)

        print ans_str
        f.write(ans_str)

