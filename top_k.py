import pandas as pd
import numpy as np
import networkx as nx
import argparse


parser = argparse.ArgumentParser(description='3DCEMA scRNA-seq top_k evaluation')
parser.add_argument('--ref_path', type=str, help='ground_truth_file')
parser.add_argument('--pred_path', type=str, help='predicted_file')
args = parser.parse_args()


trueEdgesDF = pd.read_csv(args.ref_path, sep=',', header=0, index_col=None)
predDF = pd.read_csv(args.pred_path, sep=',', header=0, index_col=None)

net = nx.DiGraph()
first_gene = trueEdgesDF['Gene1']
second_gene = trueEdgesDF['Gene2']
k = trueEdgesDF.shape[0]
for i in range(k):
    net.add_node(first_gene[i])
    net.add_node(second_gene[i])
    net.add_edge(first_gene[i], second_gene[i])


new_predDF = predDF.sort_values(by=['EdgeWeight'], ascending=False)
new_predDF.reset_index(drop=True, inplace=True)
pred_first = new_predDF['Gene1']
pred_second = new_predDF['Gene2']

count = 0
m = min(k, new_predDF.shape[0])
for i in range(m):
    if net.has_edge(pred_first[i], pred_second[i]):
        count += 1
print((count+0.0)/m)






