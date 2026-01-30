import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

def highpass(data:np.ndarray | list, fs:int, Wn:float, order:int = 4):
    """
    This Python function implements a highpass filter using a Butterworth filter design.
    
    :param data: The `data` parameter is expected to be a one-dimensional NumPy array or a list
    containing the input signal data that you want to filter using a high-pass Butterworth filter
    :type data: np.ndarray | list
    :param fs: The `fs` parameter represents the sampling frequency of the data. It is the number of
    samples taken per second when the data was recorded or measured. It is typically measured in Hertz
    (Hz)
    :type fs: int
    :param Wn: The `Wn` parameter in the `highpass` function represents the normalized cutoff frequency
    of the highpass filter. It is a float value between 0 and 1, where 1 corresponds to the Nyquist
    frequency (half the sampling rate `fs`). This parameter determines at what frequency the
    :type Wn: float
    :param order: The `order` parameter in this function represents the order of the Butterworth filter
    to be used for the high-pass filter. It determines the sharpness of the filter's roll-off curve. A
    higher order will result in a steeper roll-off but may introduce more phase distortion, defaults to
    4
    :type order: int (optional)
    :return: The function `highpass` returns the highpass filtered data using a Butterworth filter with
    the specified parameters. The filtered data is obtained by applying a zero-phase forward and reverse
    digital filtering using the second-order sections (SOS) representation of the filter.
    """
    data = data - data[0]
    sos = sig.butter(N=order, Wn=Wn, btype='highpass', output='sos', fs=fs)
    return sig.sosfiltfilt(sos, data)
def lowpass(data:np.ndarray | list, fs:int, Wn:float, order:int = 4):
    """
    The function `lowpass` applies a Butterworth lowpass filter to input data.
    
    :param data: The `data` parameter is the input signal that you want to filter using a lowpass
    Butterworth filter. It can be either a NumPy array or a list containing the signal data points
    :type data: np.ndarray | list
    :param fs: The `fs` parameter represents the sampling frequency of the data. It is the number of
    samples per second taken from a continuous signal to represent it in a discrete form
    :type fs: int
    :param Wn: The `Wn` parameter in the `lowpass` function represents the normalized cutoff frequency
    of the lowpass filter. It is a scalar or length-2 sequence giving the critical frequencies. For a
    lowpass filter, `Wn` should be a scalar value between 0 and 1,
    :type Wn: float
    :param order: The `order` parameter in the function `lowpass` specifies the order of the Butterworth
    filter to be used for low-pass filtering. It determines the sharpness of the cutoff frequency in the
    filter response. A higher order filter will have a steeper roll-off but may introduce more phase
    distortion, defaults to 4
    :type order: int (optional)
    :return: The function `lowpass` returns the filtered data using a Butterworth lowpass filter with
    the specified parameters.
    """
    sos = sig.butter(N=order, Wn=Wn, btype='lowpass', output='sos', fs=fs)
    return sig.sosfiltfilt(sos, data)
def bandpass(data:np.ndarray | list, fs:int, Wn:list[float,float], order:int = 2, freqResponse:bool=False):
    """
    The function `bandpass` applies a Butterworth bandpass filter to input data and optionally plots the
    frequency response.
    
    :param data: The `data` parameter is the input signal that you want to filter using a bandpass
    filter. It can be provided as a NumPy array or a list containing the signal data points
    :type data: np.ndarray | list
    :param fs: The `fs` parameter represents the sampling frequency of the data. It is the number of
    samples obtained in one second
    :type fs: int
    :param Wn: The `Wn` parameter in the `bandpass` function represents the passband frequencies for the
    bandpass filter. It is a list containing two float values that define the lower and upper cutoff
    frequencies of the filter's passband
    :type Wn: list[float,float]
    :param order: The `order` parameter in the `bandpass` function specifies the order of the
    Butterworth filter to be used for bandpass filtering the input data. A higher order filter will have
    a steeper roll-off but may introduce more phase distortion. It determines how many poles the filter
    will have and affects, defaults to 2
    :type order: int (optional)
    :param freqResponse: The `freqResponse` parameter in the `bandpass` function is a boolean flag that
    determines whether to plot the frequency response of the bandpass filter. If `freqResponse` is set
    to `True`, the function will generate a plot showing the frequency response of the bandpass filter.
    If `, defaults to False
    :type freqResponse: bool (optional)
    :return: The function `bandpass` returns the filtered data after applying a bandpass filter using
    the specified parameters.
    """
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
    """
    The function `getDeltaBand` filters input data to extract the surface EEG signal in the 0.4-4Hz
    frequency range.
    
    :param data: The `data` parameter is expected to be a NumPy array containing the surface EEG data
    that you want to process for the Delta band (0.4-4Hz) frequency range
    :type data: np.ndarray
    :param fs: The parameter `fs` represents the sampling frequency of the data. It is the number of
    samples obtained in one second
    :param order: The `order` parameter in the `getDeltaBand` function specifies the order of the filter
    to be used for both the highpass and lowpass filtering operations applied to the input data. A
    higher order filter will have a steeper roll-off and potentially better stopband attenuation, but it
    may also, defaults to 4 (optional)
    :return: The function `getDeltaBand` returns the input data after applying a highpass filter at
    0.4Hz and a lowpass filter at 4Hz with the specified order.
    """
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
    """
    The function `getAlphaBand` filters input data to extract surface EEG signals in the 8-13Hz
    frequency range.
    
    :param data: The `data` parameter is expected to be a NumPy array containing the surface EEG data
    that you want to filter for the 8-13Hz alpha band
    :type data: np.ndarray
    :param fs: The parameter `fs` represents the sampling frequency of the data in Hz. It is used in the
    highpass and lowpass filter functions to specify the sampling frequency of the data
    :param order: The `order` parameter in the `getAlphaBand` function is used to specify the order of
    the filter for both the highpass and lowpass operations applied to the input data. In signal
    processing, the order of a filter determines the number of previous inputs that will be considered
    when calculating the current, defaults to 4 (optional)
    :return: The function `getAlphaBand` returns the surface EEG data filtered between 8-13Hz after
    applying both highpass and lowpass filters with the specified order.
    """
    """Surface EEG 8-13Hz"""
    data = highpass(data,fs=fs,Wn=8,order=order)
    data = lowpass(data,fs=fs,Wn=13,order=order)
    return data
    
def getBetaBand(data:np.ndarray, fs,order = 4):
    """
    The function `getBetaBand` filters input data to extract the surface EEG signal in the 14-30Hz
    frequency range.
    
    :param data: The `data` parameter is expected to be a NumPy array containing the surface EEG data
    that you want to filter within the 14-30Hz beta band range
    :type data: np.ndarray
    :param fs: The parameter `fs` represents the sampling frequency of the data. It is the number of
    samples obtained in one second
    :param order: The `order` parameter in the `getBetaBand` function is used to specify the order of
    the filter to be applied during the highpass and lowpass operations on the input data. In signal
    processing, the order of a filter determines the number of previous inputs that will be considered
    when calculating the, defaults to 4 (optional)
    :return: The function `getBetaBand` returns the surface EEG data filtered between 14-30Hz after
    applying a highpass filter at 13Hz and a lowpass filter at 30Hz.
    """
    """Surface EEG 14-30Hz"""
    data = highpass(data,fs=fs,Wn=13,order=order)
    data = lowpass(data,fs=fs,Wn=30,order=order)
    return data
    
def getGammaBand_EEG(data:np.ndarray, fs,order = 4):
    """
    This function filters EEG data to extract the gamma band frequencies (30-55 Hz).
    
    :param data: The `data` parameter is expected to be a NumPy array containing the surface EEG data
    that you want to filter for the gamma band frequency range (30-55 Hz)
    :type data: np.ndarray
    :param fs: The parameter `fs` represents the sampling frequency of the EEG data. It is the number of
    samples obtained in one second
    :param order: The `order` parameter in the `getGammaBand_EEG` function is used to specify the order
    of the filter to be applied during highpass and lowpass filtering operations on the EEG data. A
    higher order filter will have a steeper roll-off and potentially better attenuation of frequencies
    outside the pass, defaults to 4 (optional)
    :return: The function `getGammaBand_EEG` returns the surface EEG data filtered between 30-55 Hz
    after applying a highpass filter at 30 Hz and a lowpass filter at 55 Hz.
    """
    """Surface EEG 30-55 Hz"""
    data = highpass(data,fs=fs,Wn=30,order=order)
    data = lowpass(data,fs=fs,Wn=55,order=order)
    return data

def getGammaBand_sEEG(data:np.ndarray, fs,order = 4):
    """
    The function `getGammaBand_sEEG` filters input data to extract the gamma band (55-115 Hz) signal in
    sEEG recordings.
    
    :param data: The `data` parameter is expected to be a NumPy array containing the signal data for
    which you want to extract the gamma band (55-115 Hz) using sEEG (stereoelectroencephalography)
    processing
    :type data: np.ndarray
    :param fs: The `fs` parameter represents the sampling frequency of the data. It is the number of
    samples taken per second when the data was recorded
    :param order: The `order` parameter in the `getGammaBand_sEEG` function refers to the order of the
    filter used for highpass and lowpass filtering operations on the input data. In signal processing,
    the order of a filter determines the number of previous inputs and outputs used to calculate the
    current output, defaults to 4 (optional)
    :return: The function `getGammaBand_sEEG` returns the input data after applying a highpass filter at
    30 Hz and a lowpass filter at 55 Hz, focusing on the gamma band frequency range of 55-115 Hz in sEEG
    data.
    """
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
    """
    The function `zscore_normalize` calculates the z-score normalization of a given numpy array data.
    
    :param data: It looks like you haven't provided the actual data for the `data` parameter in the
    `zscore_normalize` function. In order to perform z-score normalization on the data, you need to pass
    an array of numerical values to the function
    :type data: np.ndarray
    :return: The function `zscore_normalize` returns the z-score normalized version of the input data
    array.
    """
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
    # This line of code is implementing a moving average filter using convolution. Here's a breakdown
    # of what it does:
    return np.convolve(data, weights, mode='same')/ window_size

def moving_average_scipy(data, window):
    """
    The function `moving_average_scipy` calculates the moving average of a given data array using the
    `uniform_filter1d` function from the `scipy.ndimage` module.
    
    :param data: Data is the input array or sequence of values for which you want to calculate the
    moving average
    :param window: The `window` parameter in the `moving_average_scipy` function represents the size of
    the moving average window. It determines the number of data points to include in the calculation of
    the moving average at each step. A larger window size will result in a smoother moving average but
    may lose some detail in
    :return: The `moving_average_scipy` function returns the moving average of the input data using a
    specified window size. It uses the `uniform_filter1d` function from the `scipy.ndimage` module to
    calculate the moving average. The result is the moving average values for the input data, with the
    edges handled based on the specified mode and origin parameters.
    """
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(data, window, mode='constant', origin=-(window//2))[:-(window-1)]

def sliceArray(array, interval):
    """
    The function `sliceArray` takes an array and an interval as input and returns a sliced portion of
    the array based on the interval provided.
    
    :param array: The `array` parameter is a list or array that you want to slice or extract a portion
    from
    :param interval: The `interval` parameter is a list containing two elements. The first element is
    the starting index of the slice, and the second element is the ending index (exclusive) of the
    slice. The `sliceArray` function takes an `array` and returns a slice of that array based on the
    specified
    :return: The function `sliceArray` takes two arguments: `array` and `interval`. It returns a slice
    of the `array` based on the `interval` provided. The slice starts at the index `interval[0]` and
    ends at the index `interval[1]`.
    """
    return array[interval[0]:interval[1]]