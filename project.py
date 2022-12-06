import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.nonparametric.smoothers_lowess import lowess
import glob
from scipy import signal
from scipy.fft import rfft, rfftfreq, fft, ifft, fftfreq
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import datetime
import matplotlib.dates as mdates
# from pykalman import KalmanFilter
col = 'aT'


# all files in data
for f in glob.glob(sys.argv[1]):
    walk_data = pd.read_csv(f, sep=',', parse_dates=['time'], names=['time', 'ax', 'ay', 'az', 'aT'], skiprows=1)
    walk_data['time'] = pd.to_datetime(walk_data.time.astype(str), format='%H:%M:%S:%f')
    last_time = walk_data.iloc[-1]['time'] - datetime.timedelta(seconds=5)
    walk_data = walk_data[(walk_data['time'] <= last_time)]

    #cut off extra unnecessary data ex: before walking the steps
    tmp = walk_data[(walk_data[col]>=1.5)]
    first_time = tmp.iloc[0]['time']
    #before , deletes rows before that time
    walk_data = walk_data[(walk_data['time'])>=first_time]
    walk_data = walk_data.reset_index()
    

    b, a = signal.butter(3, 0.02, btype='lowpass', analog=False)
    lowpass = signal.filtfilt(b, a, walk_data[col])
    peak_heights = 1.5
    peaks, _ = signal.find_peaks(lowpass, height=peak_heights)

    # fourier transform
    # yf = rfft(lowpass)
    # xf = rfftfreq(len(walk_data.index), 1/205)

    # plt.figure(figsize=(12,4))
    # plt.plot(xf/60, np.abs(yf))
    # print(f'output{f[4:-4]}fft.png')
    # plt.savefig(f'output{f[4:-4]}fft.png')


    # trying something new
    # https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
    # X = fft(lowpass)
    # print(X)
    # N = len(X)
    # n = np.arange(N)

    # sr = 205 #ex: 1/3600
    # T = N/sr
    # freq = n/T

    # n_oneside = N//2
    # f_oneside = freq[:n_oneside]
    f_oneside = rfftfreq(N, 1/205)[:N//2]
    plt.figure(figsize = (12, 6))
    plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.savefig(f'output{f[4:-4]}fft.png')

    # theres 2 peaks

    # period
    t_h = 1/ f_oneside

    plt.figure(figsize=(12,6))
    plt.plot(t_h, np.abs(X[:n_oneside])/n_oneside)
    # plt.xticks([12, 24, 84, 168])
    # plt.xlim(0, 12)
    plt.xlabel('Period ($hopefully seconds$)')
    plt.savefig(f'output{f[4:-4]}fft.png')



    plt.title("Walk")
    plt.xlabel("Time")
    plt.ylabel("Total acceleration")
    plt.figure(figsize=(12,4))

    # plt.plot(walk_data['time'], poly[walk_data['time']])

    plt.plot(walk_data["time"], walk_data[col], 'b.', markersize=1)
    plt.plot(walk_data["time"], lowpass, 'r-')

    plt.plot(walk_data['time'][peaks], lowpass[peaks], "x", color='magenta', markersize=8)
    plt.plot(walk_data['time'], np.zeros_like(lowpass)+peak_heights, "--", color="gray")
    plt.savefig(f'output{f[4:-4]}.png')


    plt.close('all')
