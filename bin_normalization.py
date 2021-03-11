import pandas as pd
import numpy as np
import math as m
import os


def build_bin_normalized_matrix(source_path):
    if os.path.exists(source_path + 'bin-normalized-matrix.csv'):
        print('Bin normalized matrix already exists.')
        return
    data = pd.read_csv(source_path + 'ExpressionData.csv', sep=',')
    print(data)
    data.set_index('Unnamed: 0', inplace=True)
    data = data.T
    gene_name = list(data)
    k = data.shape[0]
    new_data = pd.DataFrame()
    for gene in gene_name:
        temp = data[gene]
        non_zero_element = []
        for n in range(0, k):
            if temp[n] != 0:
                non_zero_element.append(m.log(temp[n]))
        mean = np.mean(non_zero_element)
        tmin = np.min(non_zero_element)
        std = np.std(non_zero_element)
        tmax = np.max(non_zero_element)
        lower_bound = max(mean - 2 * std, tmin)
        upper_bound = min(mean + 2 * std, tmax)
        bucket_width = (upper_bound - lower_bound) / 16
        for n in range(0, k):
            if temp[n] != 0:
                index = m.floor((m.log(temp[n]) - lower_bound) / bucket_width)
                if index >= 16:
                    index = 15
                if index < 0:
                    index = 0
            else:
                index = 0
            data[gene][n] = index
        new_data[gene] = data[gene]
    print(new_data)
    file = source_path + 'bin-normalized-matrix.csv'
    new_data.to_csv(file, index=False)