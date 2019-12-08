from bs4 import BeautifulSoup
import pandas as pd
import requests
import numpy as np

news_tag = {
    "세계일보": "#article_txt",
    "중앙일보": "#article_body",
    "한겨례": ".article-text",
    "국민일보": "#articleBody",
    "동아일보": "article_txt",
    "서울신문": "#atic_txt1",
    "조선일보": "#news_body_id"
}

if __name__ == '__main__':
    news = pd.read_csv("./2019_사회/뉴스데이터_20190912-20191012.csv", converters={"뉴스 식별자": str})
    news = news.set_index("뉴스 식별자")
    news = news[news["분석제외 여부"].isna()]
    date_list = set(news["일자"].tolist())

    url_matrix = news[["일자", "언론사", "통합 분류1", "URL"]].values

    article_list = list()

    for i, row in enumerate(url_matrix):
        url = row[-1]
        company = row[1]
        if i % 100 == 0:
            print(row)
        if company in news_tag:
            try:
                contents_key = news_tag[company]
                req = requests.get(url)
                cont = req.content
                soup = BeautifulSoup(cont, 'html.parser')
                if soup.select_one(contents_key) is not None:
                    buffer = list()
                    for script in soup.find_all('script'):
                        script.extract()
                    article = soup.select_one(contents_key).get_text().replace("\n", " ").replace("\r", "")
                    article_list.append(row.tolist() + [article])

            except Exception as ex:
                print(ex)
                continue

    np.save("./뉴스데이터_20190912-20191012.npy", article_list)