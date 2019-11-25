import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # 1. 한 달 동안 꾸준한 빈도 수로 발생한 단어 = 이벤트가 아님
    startDate = datetime.strptime("20190912", "%Y%m%d")
    endDate = datetime.strptime("20191012", "%Y%m%d")

    date_array = (datetime.strftime(startDate + timedelta(days=x),  "%Y%m%d") for x in range(0, (endDate - startDate).days))

    # 1.1 일간 빈도수 파일을 전부 읽은 다음 통합
    df_list = list()
    filename = "./2019_사회/combine_counter_%s.csv"
    for date in date_array:
        df = pd.read_csv(filename % date)
        df_list.append(df)

    # 1.2 pivot (행 : 일자 / 열 : 문자)
    concat_df = pd.DataFrame(pd.concat(df_list))
    pivot = concat_df.pivot_table(values="문서 식별자", index="일자", columns='Combination')
    pivot.to_hdf("./pivot.hdf", "count_matrix")

    # pivot.diff().to_hdf("./diff1.hdf", "diff_matrix")
