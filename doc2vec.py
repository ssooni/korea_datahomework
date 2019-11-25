from gensim.models import Doc2Vec
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

            doc_list.append(TaggedDocument(words=news_list[i][-1].split(), tags=[news_list[i][2] + '%s' % i]))
        except Exception as ex:
            print(i, ex)

    doc2vec_model = Doc2Vec(doc_list, vector_size=300)
    doc2vec_model.save("./doc2vec.model")