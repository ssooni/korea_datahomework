import pandas as pd
import re
import numpy as np
from itertools import combinations

# 2개 단어로 조합
def combine(batch):
    return pd.Series(list(combinations(set(batch), 2)))

def combine_counter(news, noOfNews, count_df, outFilePath):
    print("combine_counter Start")
    news["combine_counter"] = news["특성추출"].apply(lambda x: combinations(x.split(","), 2))
    date_list = news["일자"].values

    # 숫자가 포함된 문자의 경우 제거
    number = re.compile("([0-9]+)")
    count_df = count_df.set_index("특성추출")

    keyword_list = news["combine_counter"].values
    keyword_index_list = []
    total_combin_list = set()
    for i, keyword in enumerate(keyword_list):
        for k in keyword:
            combine_keyword = "_".join(k)
            number_count = len(number.findall(combine_keyword))
            if number_count == 0:
                value1 = count_df.loc[k[0], "Value"]
                value2 = count_df.loc[k[1], "Value"]
                keyword_index_list.append([news.index[i], date_list[i], "_".join(k), value1 + value2])
                total_combin_list.add("_".join(k))

        if i % 500 == 0:
            print(i)

    df = pd.DataFrame(keyword_index_list, columns=["문서 식별자", "일자", "Combination", "Value"])

    combine_df = df.groupby(["일자", "Combination"]).count().reset_index()[["일자", "Combination","문서 식별자"]]
    values_df = df[["일자", "Combination", "Value"]].drop_duplicates(["일자", "Combination"])
    combine_df = pd.DataFrame(combine_df.merge(values_df, on=["일자", "Combination"], how="inner"))
    print(combine_df.head())
    combine_df["Lift"] = (np.log(combine_df["문서 식별자"]) -  np.log(noOfNews) - combine_df["Value"])
    combine_df["targetValue"] = combine_df["Lift"] * combine_df["문서 식별자"]
    combine_df.sort_values("targetValue", ascending=False).to_csv(outFilePath, index=False)
    print("combine_counter End : " + str(len(total_combin_list)) + " combine words")
    return total_combin_list

def counter(news, noOfNews, outFilePath):
    print("counter Start")
    news["counter"] = news["특성추출"].apply(lambda x: x.split(","))
    date_list = news["일자"].values
    # 숫자가 포함된 문자의 경우 제거
    number = re.compile("([0-9]+)")

    keyword_list = news["counter"].values
    keyword_index_list = []
    total_keyword_list = set()
    for i, keyword in enumerate(keyword_list):
        for k in keyword:
            number_count = len(number.findall(k))
            if number_count == 0:
                keyword_index_list.append([news.index[i], date_list[i], k])
                total_keyword_list.add(k)
    df = pd.DataFrame(keyword_index_list, columns=["문서 식별자", "일자", "특성추출"])

    count_df = df.groupby(["일자", "특성추출"]).count().reset_index()
    count_df["Value"] = np.log(count_df["문서 식별자"].astype(float)) - np.log(noOfNews)
    count_df.sort_values("특성추출", ascending=False).to_csv(outFilePath, index=False)
    print("counter End : " + str(len(total_keyword_list)) + " words")

    return count_df

if __name__ == '__main__':
    news = pd.read_csv("./2019_사회/뉴스데이터_20190912-20191012.csv", converters={"뉴스 식별자": str})
    news = news.set_index("뉴스 식별자")
    news = news[news["분석제외 여부"].isna()]
    date_list = set(news["일자"].tolist())
    print(news.head())
    for date in date_list:
        daliy_news = news[news["일자"] == date]
        noOfNews = len(daliy_news.index)
        print("[" + str(date) + "] the number of news : " + str(noOfNews))

        count_df = counter(daliy_news,noOfNews, "./2019_사회/counter_%s.csv" % str(date))
        combine_counter(daliy_news, noOfNews, count_df, "./2019_사회/combine_counter_%s.csv" % str(date))