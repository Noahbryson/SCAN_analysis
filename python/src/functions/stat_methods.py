import numpy as np
import scipy.stats as stats

def calc_rmse(a,b):
    N = len(a)
    rmse = np.sqrt(np.sum((a-b)**2)/N)
    return rmse


def euclidean_distance(data:np.ndarray | list):
    return np.sqrt(np.sum(np.square(data)))


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

def signed_cross_correlation(m:float or np.ndarray,r:float or np.ndarray,num_r=10,num_m=10): # type: ignore
    """
    
    """
    # rsq = metrics.r2_score(m,r)
    N = (num_r*num_m)/((num_r+num_m)**2)
    m_in = np.mean(m)
    r_in = np.mean(r)
    variance = np.var([m,r])
    res = (m_in - r_in)**3 / (abs(m_in-r_in)*variance) * N
    return res



# TODO: Implement the other statistical measures in stats module for use here. 
def calc_ROC(a,b,plot=False):
    from sklearn.metrics import roc_curve,roc_auc_score
    import matplotlib.pyplot as plt
    """
    ----------
    Parameters
    ----------
    a: array-like
        experimental distribution to test
    b: array-like
        baseline distribution to test
    """
    an = len(a)
    bn = len(b)
    aLab = [1 for _ in range(an)]
    bLab = [0 for _ in range(bn)]
    scores = list(a)
    scores.extend(b)
    lab = aLab
    lab.extend(bLab)
    fpr, tpr, thresh = roc_curve(lab,scores)
    auc = roc_auc_score(lab,scores)
    if plot:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC)\nAUC:{auc}')
        plt.legend(loc="lower right")
        plt.show()
    return auc

def biserialSpearmanCorrelation(a,b):
    # implementation of this will require broadband gamma timeseries power
    return 0

def confidence_interval(a, alpha: float=0.95):
    if len(np.shape(a)) == 1:
        CI = stats.t.interval(confidence=alpha, df = len(a)-1,
                loc=np.mean(a),scale=stats.sem(a))
        return CI
    else:
        a = np.transpose(a)
        output = np.zeros((len(a),2))
        for i,j in enumerate(a):
            CI = stats.t.interval(confidence=alpha, df = len(j)-1,
                    loc=np.mean(j),scale=stats.sem(j))
            output[i] = CI
        return output

