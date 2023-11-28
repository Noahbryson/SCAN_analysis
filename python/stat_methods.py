import numpy as np
import scipy.stats as stats


def calc_rmse(a,b):
    N = len(a)
    rmse = np.sqrt(np.sum((a-b)**2)/N)
    return rmse

def calc_rSquared(a,b):
    RSS = np.sum((a-b)**2)
    TSS = np.sum((a-np.mean(a))**2)
    r_sq = 1 - (RSS/TSS)
    return r_sq

def calc_cosine_similarity(a,b):
    num = np.dot(a,b)
    a2 = np.sqrt(sum(a**2))
    b2 = np.sqrt(sum(b**2))
    res = num  / (a2*b2)
    return res

def calc_spearman(a,b):
    res,p = stats.spearmanr(a=a,b=b)
    return res, p

def calc_pearson(a,b):
    res,p = stats.pearsonr(x=a,y=b)
    return res, p