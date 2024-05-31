import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

def highpass(data:np.ndarray | list, fs:int, Wn:float, order:int = 4):
    data = data - data[0]
    sos = sig.butter(N=order, Wn=Wn, btype='highpass', output='sos', fs=fs)
    return sig.sosfiltfilt(sos, data)
def lowpass(data:np.ndarray | list, fs:int, Wn:float, order:int = 4):
    sos = sig.butter(N=order, Wn=Wn, btype='lowpass', output='sos', fs=fs)
    return sig.sosfiltfilt(sos, data)
def bandpass(data:np.ndarray | list, fs:int, Wn:list[float,float], order:int = 2, freqResponse:bool=False):
    data = data - data[0]
    sos = sig.butter(N=order, Wn=Wn, btype='bandpass', output='sos', fs=fs)
    if freqResponse:
        w,h = sig.sosfreqz(sos,fs=fs)
        db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
        plt.figure('bandpass freq resp')
        plt.title(f'{order}nd order bp\n {Wn[0]}-{Wn[-1]} Hz')
        plt.semilogx(w,db)
        plt.show()
    return sig.sosfiltfilt(sos, data)
def notch(data:np.ndarray | list, fs:int, Wn:float, Q:int, order:int = 4):
    """
    Parameters
    ----------
    data: array-like
        data to be notched
    w0 : float
        Frequency to remove from a signal. If fs is specified, this is in the same units as fs. By default, it is a normalized scalar that must satisfy 0 < w0 < 1, with w0 = 1 corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes notch filter -3 dB bandwidth bw relative to its center frequency, Q = w0/bw.
    fs : float, optional
        The sampling frequency of the digital system
    """
    b,a = sig.iirnotch(Wn, Q, fs)
    for _ in range(order):
        data = sig.filtfilt(b,a,data)
    return data

def getDeltaBand(data:np.ndarray, fs, order = 4):
        """Surface EEG 0.4-4Hz"""
        data = highpass(data,fs=fs,Wn=0.4,order=order)
        data = lowpass(data,fs=fs,Wn=4,order=order)
        return data
    
def getThetaBand(data:np.ndarray, fs,order = 4):
    """Surface EEG 4-8Hz"""
    data = highpass(data,fs=fs,Wn=4,order=order)
    data = lowpass(data,fs=fs,Wn=8,order=order)
    return data

def getAlphaBand(data:np.ndarray, fs,order = 4):
    """Surface EEG 8-13Hz"""
    data = highpass(data,fs=fs,Wn=8,order=order)
    data = lowpass(data,fs=fs,Wn=13,order=order)
    return data
    
def getBetaBand(data:np.ndarray, fs,order = 4):
    """Surface EEG 14-30Hz"""
    data = highpass(data,fs=fs,Wn=13,order=order)
    data = lowpass(data,fs=fs,Wn=30,order=order)
    return data
    
def getGammaBand_EEG(data:np.ndarray, fs,order = 4):
    """Surface EEG 30-55 Hz"""
    data = highpass(data,fs=fs,Wn=30,order=order)
    data = lowpass(data,fs=fs,Wn=55,order=order)
    return data

def getGammaBand_sEEG(data:np.ndarray, fs,order = 4):
    """sEEG 55-115 Hz"""
    data = highpass(data,fs=fs,Wn=30,order=order)
    data = lowpass(data,fs=fs,Wn=55,order=order)
    return data


def hilbert_env(a, smooth= 0):
    """
    Computes positive real envelope with indicated smoothing with Savitsky-Golay Filter
    
    Parameters
    ----------
    a: 1-D array
        data to compute envelope of
    smooth: 0 or odd integer (Default = 0)
        number of samples to include in smoothing window
    
    Returns
    ----------
    env: Ndarray, shape of (a)
        positive real envelope with indicated smoothing 
    """
    a = a - np.mean(a)
    env = abs(sig.hilbert(a))
    if smooth:
        env = savitzky_golay(env,smooth,1)
    return env

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    ----------
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    ----------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    ----------
    Adapted from https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    ----------"""
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def zscore_normalize(data:np.ndarray):
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean)/std

def moving_average_np(data, window_size):
    """Computes the moving average of the input data using a specified window size.

    Args:
        data (numpy array): The input data to filter.
        window_size (int): The size of the moving average window.

    Returns:
        numpy array: The filtered data as a moving average.
    """
    weights = np.ones(window_size) 
    return np.convolve(data, weights, mode='same')/ window_size

def moving_average_scipy(data, window):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(data, window, mode='constant', origin=-(window//2))[:-(window-1)]

def sliceArray(array, interval):
    return array[interval[0]:interval[1]]