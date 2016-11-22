from gensim.models import Word2Vec

def train(docs, output=None, **kwargs):
    model = Word2Vec(docs, **kwargs)
    if output != None:
        if output==1:
            output='vecmodel.cpk'
        model.save(output)
    return model
