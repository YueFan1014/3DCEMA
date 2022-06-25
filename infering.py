import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import model
import torch
import pandas as pd
import networkx as nx
import numpy as np
import bin_normalization
import datetime


parser = argparse.ArgumentParser(description='3DCEMA Infering')
parser.add_argument('--num_of_classes', default=3, type=int, help='num of classes')
parser.add_argument('--data_directory', type=str, help='infering directory')
parser.add_argument('--model_name', type=str, help='model name')
args = parser.parse_args()


def infering_in_silico(path, model_name, batch_size=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = model.resnet10(sample_size=16, sample_duration=16, num_classes=3)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    checkpoint = torch.load(model_name)
    net.load_state_dict(checkpoint['net'])

    normalized_matrix = pd.read_csv(path + 'bin-normalized-matrix.csv', sep=',')
    genes = list(normalized_matrix)
    gene_num = len(genes)
    cell_size = normalized_matrix.shape[0]
    graph = nx.DiGraph()

    matrix_batch = []
    name_batch = []
    num = 0
    it = 0
    start = datetime.datetime.now()

    for gene_x in genes:
        for gene_y in genes:
            for gene_z in genes:
                #print(gene_x, gene_y, gene_z)
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

                it += 1
                num += 1
                matrix = matrix.reshape((1, 16, 16, 16))
                matrix_batch.append(matrix)
                name_batch.append((gene_x, gene_y, gene_z))
                if num == batch_size or it == gene_num**3:
                    print(it)
                    end = datetime.datetime.now()
                    print(end-start)
                    start = datetime.datetime.now()
                    matrix_batch = torch.tensor(matrix_batch)
                    inputs = matrix_batch.to(device)
                    outputs = net(inputs.float())
                    _, predicted = outputs.max(1)
                    print('predicted')
                    print(predicted)
                    for i in range(num):
                        gene_x, gene_y, gene_z = name_batch[i]
                        # print(gene_x, gene_y, gene_z)
                        # print(predicted[i])
                        if predicted[i] == 1 or predicted[i] == 2:
                            if not graph.has_edge(gene_x, gene_y):
                                graph.add_edge(gene_x, gene_y)
                                graph[gene_x][gene_y]['weight'] = 0
                            if not graph.has_edge(gene_x, gene_z):
                                graph.add_edge(gene_x, gene_z)
                                graph[gene_x][gene_z]['weight'] = 0
                            if predicted[i] == 1:
                                graph[gene_x][gene_y]['weight'] += 1
                                graph[gene_x][gene_z]['weight'] -= 1
                            else:
                                graph[gene_x][gene_y]['weight'] -= 1
                                graph[gene_x][gene_z]['weight'] += 1
                    num = 0
                    matrix_batch = []
                    name_batch = []

    file = open(path + 'rankedEdges.csv', 'w')
    file.write('Gene1,Gene2,EdgeWeight\n')
    for gene_x in genes:
        for gene_y in genes:
            if graph.has_edge(gene_x, gene_y):
                text = gene_x + ',' + gene_y + ',' + str(graph[gene_x][gene_y]['weight']) + '\n'
                file.write(text)
    file.close()


def infering_real(path, model_name, batch_size=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = model.resnet10(sample_size=16, sample_duration=16, num_classes=2)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    checkpoint = torch.load(model_name)
    net.load_state_dict(checkpoint['net'])

    normalized_matrix = pd.read_csv(path + 'bin-normalized-matrix.csv', sep=',')
    genes = list(normalized_matrix)
    gene_num = len(genes)
    cell_size = normalized_matrix.shape[0]
    graph = nx.DiGraph()
    file = open(path+args.model_name+'rankedEdges.csv', 'w')
    file.write('Gene1,Gene2,EdgeWeight\n')

    matrix_batch = []
    name_batch = []
    num = 0
    it = 0
    start = datetime.datetime.now()
    for gene_x in genes:
        for gene_y in genes:
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

            it += 1
            num += 1
            matrix = matrix.reshape((1, 16, 16, 16))
            matrix_batch.append(matrix)
            name_batch.append((gene_x, gene_y))
            if num == batch_size or it == gene_num**2:
                print(it)
                end = datetime.datetime.now()
                print(end - start)
                matrix_batch = torch.tensor(matrix_batch)
                inputs = matrix_batch.to(device)
                outputs = net(inputs.float())
                _, predicted = outputs.max(1)
                #print('predicted')
                #print(outputs)
                for i in range(num):
                    gene_x, gene_y = name_batch[i]
                    text = gene_x + ',' + gene_y + ',' + str(outputs[i, 1].item() - outputs[i, 0].item()) + '\n'
                    #text = gene_x + ',' + gene_y + ',' + str(predicted[i].item()) + '\n'
                    file.write(text)
                num = 0
                matrix_batch = []
                name_batch = []
    file.close()


model_name = r'trained_models\\'+args.model_name
bin_normalization.build_bin_normalized_matrix(args.data_directory)
if args.num_of_classes == 3:
    infering_in_silico(path=args.data_directory, model_name=model_name)
else:
    infering_real(path=args.data_directory, model_name=model_name)
