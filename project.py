import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.nonparametric.smoothers_lowess import lowess
import glob
from scipy import signal
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import datetime
import matplotlib.dates as mdates
# from pykalman import KalmanFilter
col = 'aT'
for f in glob.glob(sys.argv[1]):
    walk_data = pd.read_csv(f, sep=',', parse_dates=['time'], names=['time', 'ax', 'ay', 'az', 'aT'], skiprows=1)
    walk_data['time'] = pd.to_datetime(walk_data.time.astype(str), format='%H:%M:%S:%f')
    last_time = walk_data.iloc[-1]['time'] - datetime.timedelta(seconds=5)
    walk_data = walk_data[walk_data['time'] <= last_time]

    b, a = signal.butter(3, 0.02, btype='lowpass', analog=False)
    lowpass = signal.filtfilt(b, a, walk_data[col])
    peak_heights = 1.5
    peaks, _ = signal.find_peaks(lowpass, height=peak_heights)

    # X = walk_data['time']
    # y = lowpass
    # model = make_pipeline(
    #     PolynomialFeatures(degree=10, include_bias=True),
    #     LinearRegression(fit_intercept=False)

    # )
    # model.fit(X, y)
    # timearr = np.array(walk_data.time.values)
    # print(lowpass.dtype)
    # # timearr = mdates.date2num(walk_data.time)
    # poly = np.poly1d(np.polyfit(timearr, lowpass, 10))

    # fourier transform
    yf = rfft(lowpass)
    xf = rfftfreq(len(walk_data.index), 205)

    plt.figure(figsize=(12,4))
    plt.plot(xf, np.abs(yf))
    plt.savefig(f'output/{f[4:-4]}fft.png')



    plt.title("Walk")
    plt.xlabel("Time")
    plt.ylabel("Total acceleration")
    plt.figure(figsize=(12,4))

    # plt.plot(walk_data['time'], poly[walk_data['time']])

    plt.plot(walk_data["time"], walk_data[col], 'b.', markersize=1)
    plt.plot(walk_data["time"], lowpass, 'r-')

    plt.plot(walk_data['time'][peaks], lowpass[peaks], "x", color='magenta', markersize=8)
    plt.plot(walk_data['time'], np.zeros_like(lowpass)+peak_heights, "--", color="gray")
    plt.savefig(f'output/{f[4:-4]}.png')
