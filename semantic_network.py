import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from community import community_louvain
from networkx.algorithms.community import greedy_modularity_communities

def network(base):

    startDate = datetime.strptime("20190912", "%Y%m%d")
    endDate = datetime.strptime("20191012", "%Y%m%d")

    date_array = (datetime.strftime(startDate + timedelta(days=x),  "%Y%m%d") for x in range(0, (endDate - startDate).days))


    for date in date_array:
        plt.figure(figsize=(30,30))
        df = pd.read_csv("2019_사회/combine_counter_%s.csv" % date)
        df["Word1"] = df["Combination"].apply(lambda x: x.split("_")[0])
        df["Word2"] = df["Combination"].apply(lambda x: x.split("_")[1])

        print(df.describe())

        freqBaseNetwork = nx.from_pandas_edgelist(df, "Word1", "Word2", base)
        print(nx.info(freqBaseNetwork))

        betCent = nx.eigenvector_centrality(freqBaseNetwork)
        betCentDf = pd.DataFrame.from_dict(betCent, orient="index", columns=["Value"]).sort_values("Value", ascending=False)
        betCentDf.to_csv("./2019_사회/network_%s_%s_사회.csv" % (date, base))

        top_100_nodes = betCentDf.index.tolist()
        print(top_100_nodes)

        G1 = freqBaseNetwork.subgraph(top_100_nodes)
        top_100_centrality = nx.eigenvector_centrality(G1)

        node_color = [20000.0 * G1.degree(v) for v in G1]
        node_size = [v * 10000 for v in top_100_centrality.values()]

        nx.draw_spring(G1, k=1, node_color=node_color,
                       node_size=node_size,
                       font_size=10, with_labels=True, font_family='AppleGothic')

        plt.axis('off')
        plt.savefig("./2019_사회/network_%s_%s_사회.png" % (date, base))

        partition = community_louvain.best_partition(G1, weight=base)
        pd.DataFrame.from_dict(partition, orient='index', columns=["community"]).sort_values("community")\
            .to_csv("./2019_사회/community_%s_%s_사회.csv" % (date, base))

        size = float(len(set(partition.values())))

        plt.close()
        plt.figure(figsize=(30,30))
        pos = nx.spring_layout(G1)
        count = 0.
        for com in set(partition.values()):
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                          if partition[nodes] == com]
            nx.draw_networkx_nodes(G1, pos, list_nodes, node_size=node_size,
                                   node_color=str(count / size))
        plt.axis('off')
        nx.draw_networkx_labels(G1, pos,  font_size=10, with_labels=True, font_family='AppleGothic')
        nx.draw_networkx_edges(G1, pos, alpha=0.5)
        plt.savefig("./2019_사회/community_%s_%s_사회.png" % (date, base))
        break


if __name__ == '__main__':
    network("Lift")
    network("조합 빈도수")
    network("targetValue")