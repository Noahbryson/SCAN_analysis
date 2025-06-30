import random
import distinctipy
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import _result_classes
from sklearn.neighbors import KernelDensity
class stat_res():
    def __init__(self,result,pvalue)->None:
        self.result = result
        self.pvalue = pvalue

def calc_rmse(a,b)->float:
    N = len(a)
    rmse = np.sqrt(np.sum((a-b)**2)/N)
    return rmse

    

def euclidean_distance(p,q)-> float:
    """euclidean_distance _summary_

    Args:
        p ndarray: a point in 2D or 3D space
        q ndarray: a point in 2D or 3D space

    Returns:
        dist : euclidean distance between p and q
    """
    return np.linalg.norm(p-q)

def kde(data):
    data = np.array(data)
    data = data.reshape(len(data),1)
    return KernelDensity(kernel='gaussian').fit(data)


def geometric_mean(pointset):
    
    N = len(pointset)
    return np.prod(pointset)**(1/N)

def xy_angle(x,y):
    """calculates positive angle from x-axis in x-y plane in degrees (0-360)
    Args:
        x (float): x-coordinate
        y (float): y-coordinate
    Returns:
        ang(float): positive angle from x-axis in x-y plane in degrees (0-360)
    """
    ang = np.arctan2(y,x) * 180 /np.pi
    if ang < 0:
        ang = 360 + ang
    return ang
def xz_angle(x,z):
    """returns the vertical angle in degrees of a point in the x-z plane. Negative if z < 0, positive otherwise.

    Args:
        x (float): x-coordinate
        z (float): z-coordinate

    Returns:
        vertical angle in degrees from x-axis in x-z plane
    """
    x=abs(x)
    return np.arctan(x/z) * 180 /np.pi

def distanceFromPositiveAngle(angle,target):
    if angle >=0:
        distance = abs(angle - target)
    else:
        distance = abs(angle + target)
    return distance

def angle_distances_3D(angles, targets:list):
    horiz = distanceFromPositiveAngle(angles[0],targets[0])
    vert = distanceFromPositiveAngle(angles[1],targets[1])
    return [horiz,vert]
def angle_3D(point):
    x = point[0]; y=point[1]; z = point[2]
    horiz = xy_angle(x,y)
    vert = xz_angle(x,z)
    return [horiz,vert]    

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
    
    
def one_samp_ttest(data,popmean=0,test_type='two-sided')->_result_classes.TtestResult:
    return stats.ttest_1samp(data,popmean=popmean,alternative=test_type)

def nonlinear_fit_permutation_test(x,y, model_function, popt, n_perm=1000)-> stat_res:
    from scipy.optimize import curve_fit
    r2_obs = compute_coeff_determ(y, model_function(x, *popt))
    r2_perm = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = np.random.permutation(y)
        try:
            popt_perm, _ = curve_fit(model_function, x, y_perm, p0=popt)
            r2_perm[i] = compute_coeff_determ(y_perm, model_function(x, *popt_perm))
        except:
            r2_perm[i] = 0  # or np.nan
    p_val = (np.sum(r2_perm >= r2_obs) + 1) / (n_perm + 1)
    
    return stat_res(r2_obs, p_val)
    
def pdist2(points:np.ndarray,cloud:np.ndarray,num_mins:int=100)->tuple:
    """
    pdist2 _summary_

    Args:
        point (np.ndarray): points of interest (m x [x,y,z]).
        cloud (np.ndarray): volume of interest (m x [x,y,z]). 
        num_mins (int): the number of minimum points to return.
            (default=1). currently deprecated
    Returns:
        tuple[np.ndarray,np.ndarray]: returns minimum distance and point cloud location of minimum distance
    """

    queries = points[:,None,:]
    volume = cloud[None,:,:]
    diffs = points[:,None,:] - cloud[None,:,:]
    dist_sqr = np.sum(diffs ** 2,axis=2)
    min_dist2 = np.sqrt(dist_sqr)
    x=np.argsort(min_dist2,axis=1)
    x_slice = np.tile(np.linspace(0,min_dist2.shape[0]-1,min_dist2.shape[0],dtype=int),[num_mins,1]).T.flatten()
    y_slice = x[:,:num_mins].flatten()
    dist_out = min_dist2[x_slice,y_slice].reshape(points.shape[0],num_mins)
    # min_dist2 = np.sqrt(np.min(dist_sqr,axis=1))
    # min_idx = np.argmin(dist_sqr,axis=1)
    min_idxs = min_dist2
    return dist_out
    
    
def compute_coeff_determ(ydata,ymodel)-> float:
    residuals = np.sort(ydata-ymodel)
    ss_total = np.sum((ydata-np.mean(ydata))**2)
    ss_residual = np.sum(residuals**2)
    return 1 - ss_residual/ss_total


def paired_two_sample(a:np.array,b:np.array,ax:plt.axes, colorPallet:list = distinctipy.get_colors(3,pastel_factor=0.5,rng=random.seed(35)), bars=False,jitter = 0.0, normalityAlpha = 0.05,plot_mean=True):
        
        import itertools
        import scipy.stats
        import numpy as np
        """
        Takes two ordered 1D arrays, computes a paired ttest between them 
        a: control group to compare to
        b: experimental group
        Assesses groups for normality
        Performs Paired T Test if both groups are normally distributed
        Performs Wilcoxon Signed Rank Test if otherwise
        ---
        Dependent on numpy, matplotlib.pyplot, itertools and scipy
        ---
        returns t statistics, pvalues and a matplotlib axes 
        """

        if len(a) != len(b):
            raise ValueError("a and b must be of equal length")

        while len(colorPallet) > 2:
            colorPallet.pop(1)
        
        
        """Assess Normality"""
        normalFlag = False
        for array in [a,b]:
                ks, ks_p = scipy.stats.ks_1samp(array,scipy.stats.norm.cdf)
                if ks_p <= normalityAlpha:
                    normalFlag = False
                    print('Non-Normal')
                else:
                    print('normal')
        if normalFlag :
            res = scipy.stats.ttest_rel(a=a,b=b)
        else:
            res = scipy.stats.wilcoxon(x=a,y=b)
        #make this value 0.05 if you want paired points offset from each other, or zero if stacked
        stdev_a = a.std()
        stdev_b = b.std()
        
        """If you want bar plots of means"""
        if bars:
            ax.bar(0,a.mean())
            ax.bar(1,b.mean())
            
        if plot_mean and not bars:  
            ax.errorbar(0,a.mean(), yerr=stdev_a,markerfacecolor = colorPallet[0],  alpha=0.5, ecolor='grey', capsize=10,marker='o',ms=10)
            ax.errorbar(1,b.mean(), yerr=stdev_b,markerfacecolor = colorPallet[1], alpha=0.5, ecolor='grey', capsize=10,marker='o',ms=10)
            
                
        
        for q,p in zip(a,b):
            noise = np.random.normal(0,jitter,1)
            x = [noise, 1+noise]
            y = [q,p]
            ax.scatter(noise,q,color=colorPallet[0])
            ax.scatter(1+noise,p,color=colorPallet[1])
            if p - q > 0:
                alpha = 0.4
            else:
                alpha = 0.2
            ax.plot(x,y, color='k', alpha = alpha)
        ax.set_xlim([-0.5,1.5])
        # ax.text(0.5,np.max([a,b],axis=-1),f'Test Stat={res.statistic}\np={res.pvalue}')
        # if res.pvalue < 0.05:
        #     ax.text(0.5,1.1*np.max([a,b],axis=-1),'*')
        return res,ax