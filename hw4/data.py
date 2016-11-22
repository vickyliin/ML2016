from nltk.tokenize import word_tokenize

def loadT(path='data'):
    if path[-1] != '/':
        path += '/'
    with open(path+'title_StackOverflow.txt', 'r') as f:
        titles = [ word_tokenize(x) for x in f ]
    return titles

def loadD(path='data'):
    if path[-1] != '/':
        path += '/'
    with open(path+'docs.txt', 'r') as f:
        sentences = [ word_tokenize(x) for x in f if x.strip != '']
    return sentences
