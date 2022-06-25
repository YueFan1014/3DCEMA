import networkx as nx
import pandas as pd
import numpy as np
import os
import random as rd
import math as m


def create_in_silico_training_matrices(path, k_limit):
    data_path = path + r'dataset\\'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    else:
        print('Matrices already exist.')
        return

    normalized_matrix = pd.read_csv(path + 'bin-normalized-matrix.csv', sep=',')
    genes = list(normalized_matrix)

    graph = nx.DiGraph()
    ground_truth = open(path + r'refNetwork.csv', 'r')
    ground_truth.readline()
    lines = ground_truth.readlines()
    for line in lines:
        items = line.strip().split(',')
        graph.add_edge(items[0], items[1])
        graph[items[0]][items[1]]['weight'] = items[2]
    ground_truth.close()

    classes = [], [], []
    label_dict = dict(
        [('00', 0), ('0+', 2), ('0-', 1), ('+0', 1), ('++', 0), ('+-', 1), ('-0', 2), ('-+', 2), ('--', 0)])

    for gene_x in genes:
        for gene_y in genes:
            for gene_z in genes:
                mark = ''
                if graph.has_edge(gene_x, gene_y):
                    mark += graph[gene_x][gene_y]['weight']
                else:
                    mark += '0'
                if graph.has_edge(gene_x, gene_z):
                    mark += graph[gene_x][gene_z]['weight']
                else:
                    mark += '0'
                class_num = label_dict[mark]
                classes[class_num].append((gene_x, gene_y, gene_z))

    num = min(len(classes[0]), len(classes[1]), len(classes[2]))
    if 3 * num > k_limit:
        num = m.ceil(k_limit / 3)
    class_zero = rd.sample(classes[0], num)
    class_one = rd.sample(classes[1], num)
    class_two = rd.sample(classes[2], num)
    classes = class_zero, class_one, class_two

    data_file = open(data_path + 'dataset.txt', 'w')
    cell_size = normalized_matrix.shape[0]
    for label in range(3):
        for gene_triple in classes[label]:
            gene_x, gene_y, gene_z = gene_triple
            matrix = np.zeros((16, 16, 16))
            for i in range(cell_size):
                x = normalized_matrix.loc[i, gene_x].astype('int')
                y = normalized_matrix.loc[i, gene_y].astype('int')
                z = normalized_matrix.loc[i, gene_z].astype('int')
                matrix[x][y][z] += 1
            for a in range(16):
                for b in range(16):
                    for c in range(16):
                        matrix[a][b][c] = matrix[a][b][c] * 4096 / cell_size
            temp_path = data_path + str(label) + r'\\'
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            np.save(temp_path+gene_x+'_'+gene_y+'_'+gene_z+'.npy', matrix)
            msg = str(label)+' '+gene_x+'_'+gene_y+'_'+gene_z+'.npy'+'\n'
            data_file.write(msg)
    data_file.close()


def create_real_data_training_matrices(path, k_limit):
    data_path = path + r'dataset\\'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    else:
        print('Matrices already exist.')
        return

    normalized_matrix = pd.read_csv(path + 'bin-normalized-matrix.csv', sep=',')

    graph = nx.DiGraph()
    ground_truth = open(path + r'refNetwork.csv', 'r')
    ground_truth.readline()
    lines = ground_truth.readlines()
    for line in lines:
        items = line.strip().split(',')
        graph.add_edge(items[0], items[1])
    ground_truth.close()
    genes = graph.nodes()

    data_path = path + r'dataset\\'
    data_file = open(data_path + 'dataset.txt', 'w')
    cell_size = normalized_matrix.shape[0]

    classes = [], []
    for gene_x in genes:
        for gene_y in genes:
            if not graph.has_edge(gene_x, gene_y):
                classes[0].append((gene_x, gene_y))
            else:
                classes[1].append((gene_x, gene_y))
    label_zero_num = min(m.ceil(k_limit*0.7), len(classes[0]))
    label_one_num = min(m.ceil(k_limit*0.3), len(classes[1]))

    class_zero = rd.sample(classes[0], label_zero_num)
    class_one = rd.sample(classes[1], label_one_num)
    classes = class_zero, class_one

    for label in range(2):
        for gene_pair in classes[label]:
            gene_x, gene_y = gene_pair
            matrix = np.zeros((16, 16, 16))
            for i in range(cell_size):
                x = normalized_matrix.loc[i, gene_x].astype('int')
                y = normalized_matrix.loc[i, gene_y].astype('int')
                z = normalized_matrix.loc[i, 'AVG'].astype('int')
                matrix[x][y][z] += 1
            for a in range(16):
                for b in range(16):
                    for c in range(16):
                        matrix[a][b][c] = matrix[a][b][c] * 4096 / cell_size
            temp_path = data_path + str(label)
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            np.save(temp_path + r'\\' + gene_x + '_' + gene_y + '.npy', matrix)
            msg = str(label) + ' ' + gene_x + '_' + gene_y + '.npy' + '\n'
            data_file.write(msg)
    data_file.close()