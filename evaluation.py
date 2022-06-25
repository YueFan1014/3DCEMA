import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product
import argparse


parser = argparse.ArgumentParser(description='3DCEMA evaluation')
parser.add_argument('--data_directory', type=str, help='evaluation directory')
parser.add_argument('--model_name', type=str, help='model name')
parser.add_argument('--silico', type=bool, help='is in-silico data?')
args = parser.parse_args()



trueEdgesDF = pd.read_csv(args.data_directory + 'refNetwork.csv',
                          sep=',', header=0, index_col=None)
predDF = pd.read_csv(args.data_directory + 'rankedEdges.csv', sep=',', header=0, index_col=None)



def computeScores(trueEdgesDF, predEdgeDF):
    possibleEdges = list(product(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]), repeat=2))

    TrueEdgeDict = {'|'.join(p): 0 for p in possibleEdges}
    PredEdgeDict = {'|'.join(p): 0 for p in possibleEdges}

    for key in TrueEdgeDict.keys():
        if len(trueEdgesDF.loc[(trueEdgesDF['Gene1'] == key.split('|')[0]) &
                               (trueEdgesDF['Gene2'] == key.split('|')[1])]) > 0:
            TrueEdgeDict[key] = 1

    for key in PredEdgeDict.keys():
        subDF = predEdgeDF.loc[(predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])]
        if len(subDF) == 0:
            PredEdgeDict[key] = 0
        else:
            if args.silico:
                PredEdgeDict[key] = abs(subDF.EdgeWeight.values[0])
            else:
                PredEdgeDict[key] = subDF.EdgeWeight.values[0]

    outDF = pd.DataFrame([TrueEdgeDict, PredEdgeDict]).T
    outDF.columns = ['TrueEdges', 'PredEdges']
    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)
    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)
    return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr)


precision, recall, FPR, TPR, AUPRC, AUROC = computeScores(trueEdgesDF, predDF)
print('AUPRC=', AUPRC, ' AUROC=', AUROC)





