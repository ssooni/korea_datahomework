from gensim.models import Doc2Vec
from gensim.models.word2vec import Word2Vec

import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument

if __name__ == '__main__':
    news_list = np.load("./뉴스데이터_20190912-20191012.npy")

    doc_list = list()
    for i in range(len(news_list)):
        try:
            if i % 100 == 0:
               print(news_list[i])

            doc_list.append(news_list[i][-1])
        except Exception as ex:
            print(i, ex)

    w2v_model = Word2Vec(doc_list, size=300)
    w2v_model.save("./word2vec.model")