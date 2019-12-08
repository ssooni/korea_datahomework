import pandas as pd
import numpy as np
import re
from itertools import combinations


def counter(news):
    print("counter Start")
    news["counter"] = news["특성추출"].apply(lambda x: x.split(","))
    date_list = news["일자"].values
    # 숫자가 포함된 문자의 경우 제거
    number = re.compile("([0-9]+)")
    stopword = pd.read_csv("2019_사회/stopword")["불용어"].tolist()

    keyword_list = news["counter"].values
    keyword_index_list = []
    total_keyword_list = set()
    for i, keyword in enumerate(keyword_list):
        for k in keyword:
            number_count = len(number.findall(k))
            if number_count == 0 and k not in stopword:
                keyword_index_list.append([news.index[i], date_list[i], k])
                total_keyword_list.add(k)
    df = pd.DataFrame(keyword_index_list, columns=["문서 식별자", "일자", "특성추출"])
    count_df = df.groupby(["일자", "특성추출"]).count().reset_index()
    count_df.columns = ["일자", "특성추출", "빈도수"]
    print("counter End : " + str(len(total_keyword_list)) + " words")
    return count_df


def combine_counter(news, count_df, min_support=0.01):
    print("combine_counter Start")
    news["combine_counter"] = news["특성추출"].apply(lambda x: combinations(x.split(","), 2))
    date_list = news["일자"].values

    # 숫자가 포함된 문자의 경우 제거
    number = re.compile("([0-9]+)")
    count_df = count_df.set_index("특성추출")

    keyword_list = news["combine_counter"].values
    keyword_index_list = []
    total_combin_list = set()
    stopword = pd.read_csv("2019_사회/stopword")["불용어"].tolist()

    for i, keyword in enumerate(keyword_list):
        for k in keyword:
            combine_keyword = "_".join(k)
            number_count = len(number.findall(combine_keyword))
            if number_count == 0 and k[0] not in stopword and k[1] not in stopword:
                value1 = count_df.loc[k[0], "빈도수"]
                value2 = count_df.loc[k[1], "빈도수"]

                support1 = count_df.loc[k[0], "support"]
                support2 = count_df.loc[k[1], "support"]

                if support1 < np.log(min_support) or support2 < np.log(min_support):
                    continue

                keyword_index_list.append([news.index[i], date_list[i], "_".join(k), value1, value2, support1, support2])
                total_combin_list.add("_".join(k))

        if i % 500 == 0:
            print(i)

    df = pd.DataFrame(keyword_index_list, columns=["문서 식별자", "일자", "Combination", "빈도수 A",
                                                   "빈도수 B", "Support A", "Support B"])
    print(df.head())
    combine_df = df.groupby(["일자", "Combination"]).count().reset_index()[["일자", "Combination", "문서 식별자"]]
    combine_df.columns = ["일자", "Combination", "조합 빈도수"]
    values_df = df[["일자", "Combination", "빈도수 A", "빈도수 B", "Support A", "Support B"]].drop_duplicates(["일자", "Combination"])
    values_df["Combination"] = values_df["Combination"].astype(str)
    combine_df["Combination"] = combine_df["Combination"].astype(str)

    combine_df = pd.DataFrame(combine_df.merge(values_df, on=["일자", "Combination"], how="inner"))

    return combine_df


def support(daily_news, noOfNews):
    # Support(B) = (Transactions containing (B))/(Total Transactions)
    # 해당 단어가 포함된 문서 수 / 전체 문서 수
    daily_news.loc[:, "support"] = np.log(daily_news.loc[:, "빈도수"].astype(float)) - np.log(noOfNews)
    return daily_news


def confidence(daily_news):
    # Confidence(A→B) = (Transactions containing both (A and B))/(Transactions containing A)
    # A와 B가 동시에 나온 문서 수 / A 가 포함된 문서 수
    daily_news.loc[:, "confidence A"] = np.log(daily_news.loc[:, "조합 빈도수"].astype(float)) - np.log(daily_news.loc[:, "빈도수 A"].astype(float))
    daily_news.loc[:, "confidence B"] = np.log(daily_news.loc[:, "조합 빈도수"].astype(float)) - np.log(daily_news.loc[:, "빈도수 B"].astype(float))

    return daily_news


def lift(daily_news):
    # Lift(A→B) = (Confidence (A→B))/(Support (B))
    # A와 B가 컨피던스 / 서포트 B
    daily_news.loc[:, "Lift"] = daily_news.loc[:, "confidence A"].astype(float) - daily_news.loc[:, "Support B"].astype(float)
    daily_news.loc[:, "targetValue"] = daily_news.loc[:, "Lift"] * daily_news.loc[:, "조합 빈도수"]

    return daily_news


if __name__ == '__main__':
    news = pd.read_csv("./2019_사회/뉴스데이터_20190912-20191012.csv", converters={"뉴스 식별자": str})
    news = news.set_index("뉴스 식별자")
    news = news[news["분석제외 여부"].isna()]
    date_list = set(news["일자"].tolist())
    print(news.head())
    for date in date_list:
        origin = news[news["일자"] == date]
        noOfNews = len(origin.index)
        print("[" + str(date) + "] the number of news : " + str(noOfNews))
        daily_news = counter(origin)
        print(daily_news.head())

        daily_news = support(daily_news, noOfNews)
        print(daily_news.head())

        daily_news = combine_counter(origin, daily_news)
        print(daily_news.head())

        daily_news = confidence(daily_news)
        print(daily_news.head())

        daily_news = lift(daily_news)
        print(daily_news.head())

        daily_news.to_csv("./2019_사회/combine_counter_%s.csv" % str(date), index=False)
