from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from konlpy.tag import Komoran
from konlpy.tag import Kkma


def doc2vec_train():
    news_list = np.load("./뉴스데이터_20190912-20191012.npy")
    tagger = Komoran()
    doc_list = list()
    for i in range(len(news_list)):
        try:
            if i % 100 == 0:
               print(news_list[i])

            doc_list.append(TaggedDocument(words=tagger.morphs(news_list[i][-1]), tags=[news_list[i][2] + '%s' % i]))
        except Exception as ex:
            print(i, ex)
    np.save("./tokenizer.npy", doc_list)
    doc2vec_model = Doc2Vec(doc_list, vector_size=300)
    doc2vec_model.save("./doc2vec_kkma.model")


def create_word_vector_matrix():
    doc2vec_model = Doc2Vec.load("./doc2vec_kkma.model")
    news_list = np.load("./뉴스데이터_20190912-20191012.npy")
    result_list = list()

    for i in range(len(news_list)):
        vector_list = list()
        try:
            if i % 100 == 0:
               print(news_list[i])

            article = news_list[i][-1].split()

            for word in article:
                if word in doc2vec_model.wv.vocab:
                    vector_list.append(doc2vec_model[word])

            if len(vector_list) > 0:
                result_list.append([news_list[i][2].split(">")[1], np.asarray(vector_list)])

        except Exception as ex:
            print(i, ex)

    np.save("./article.npy", result_list)


if __name__ == '__main__':
    doc2vec_train()
    # create_word_vector_matrix()
