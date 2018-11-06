import numpy as np
from scipy.special import comb

def external_index(v1, v2):
    TP, FN, FP, TN = confusion_index(v1, v2)
    RI = (TP + TN) / (TP + FN + FP + TN);
    ARI = 2 * (TP * TN - FN * FP) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN));
    JI = TP / (TP + FN + FP);
    FM = TP / np.sqrt((TP + FN) * (TP + FP));
    return RI, ARI, JI, FM

def confusion_index(v1, v2):
    cmatrix = contingency(v1, v2)
    size = np.size(v1)
    sum_rows = np.sum(cmatrix, 0)
    sum_cols = np.sum(cmatrix, 1)
    N = comb(size, 2)
    TP = np.sum(list(map(lambda x: comb(x, 2), cmatrix)))
    FN = np.sum(list(map(lambda x: comb(x, 2), sum_rows))) - TP
    FP = np.sum(list(map(lambda x: comb(x, 2), sum_cols))) - TP
    TN = N - TP - FN - FP
    return TP, FN, FP, TN

def contingency(v1, v2):
    res = np.zeros((np.max(v1), np.max(v2)))
    for i in range(0, np.size(v1)):
        res[v1[i] - 1, v2[i] - 1] = res[v1[i] - 1, v2[i] - 1] + 1
    return res
