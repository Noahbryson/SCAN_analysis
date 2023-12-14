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

def mannwhitneyU(a,b):
    res = stats.mannwhitneyu(a,b)
    return res

def cohendsD(a, b):
    """ compute cohen's D on two samples. Positive d indicates a > b, negative indicates b > a
        method with degrees of freedom modulation is used when the number of samples in each distribution is different
        ----------
        Parameters
        ----------
        a: array-like
            experimental distribution to test
        b: array-like
            baseline distribution to test
        ----------
        Returns
        ----------
        d: float
            Cohen's D of the two distributions"""
    nx = len(a)
    ny = len(b)
    if nx != ny:
        dof = nx + ny - 2
        return (np.mean(a) -np.mean(b)) / np.sqrt(((nx-1)*np.std(a, ddof=1) ** 2 + (ny-1)*np.std(b, ddof=1) ** 2) / dof)
    else:
        m1 = np.mean(a)
        m2 = np.mean(b)
        std1 = np.std(a)
        std2 = np.std(b)
        stdPooled = np.sqrt((std1**2+std2**2)/2)
        return (m1-m2)/stdPooled

# TODO: Implement the other statistical measures in stats module for use here. 
def ROC(a,b):
    return 0

def biserialSpearmanCorrelation(a,b):
    # implementation of this will require broadband gamma timeseries power
    return 0




