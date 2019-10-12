import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

def network():
    pass


if __name__ == '__main__':
    startDate = datetime.strptime("20190912", "%Y%m%d")
    endDate = datetime.strptime("20191012", "%Y%m%d")

    date_array = (datetime.strftime(startDate + timedelta(days=x),  "%Y%m%d") for x in range(0, (endDate - startDate).days))

    for date in date_array:
        df = pd.read_csv("2019_사회/combine_counter_%s.csv" % date)
        df["Word1"] = df["Combination"].apply(lambda x: x.split("_")[0])
        df["Word2"] = df["Combination"].apply(lambda x: x.split("_")[1])
        print(df[df["문서 식별자"] > 2].describe())
        freqBaseNetwork = nx.from_pandas_edgelist(df[df["문서 식별자"] > 2], "Word1", "Word2", "Lift")
        print(nx.info(freqBaseNetwork))

        betCent = nx.eigenvector_centrality(freqBaseNetwork)
        pd.DataFrame.from_dict(betCent, orient="index", columns=["Value"]).sort_values("Value", ascending=False).to_csv("./2019_사회/network_%s_사회.csv" % date)
    # pos = nx.spring_layout(freqBaseNetwork)
    # node_color = [20000.0 * freqBaseNetwork.degree(v) for v in freqBaseNetwork]
    # node_size = [v * 10000 for v in betCent.values()]
    # plt.figure(figsize=(20, 20))
    # nx.draw_networkx(freqBaseNetwork, pos=pos, with_labels=False,
    #                  node_color=node_color,
    #                  node_size=node_size)
    # plt.axis('off')
    # plt.show()
