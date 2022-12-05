import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.nonparametric.smoothers_lowess import lowess
import glob
from scipy import signal
from scipy.fft import rfft, rfftfreq
import datetime
# from pykalman import KalmanFilter
col = 'aT'
for f in glob.glob(sys.argv[1]):
    walk_data = pd.read_csv(f, sep=',', parse_dates=['time'], names=['time', 'ax', 'ay', 'az', 'aT'], skiprows=1)
    walk_data['time'] = pd.to_datetime(walk_data.time.astype(str), format='%H:%M:%S:%f')
    last_time = walk_data.iloc[-1]['time'] - datetime.timedelta(seconds=5)
    walk_data = walk_data[walk_data['time'] <= last_time]

    b, a = signal.butter(3, 0.02, btype='lowpass', analog=False)
    lowpass = signal.filtfilt(b, a, walk_data[col])

    # fourier transform
    yf = rfft(lowpass)
    xf = rfftfreq(len(walk_data.index), 1/205)

    plt.figure(figsize=(12,4))
    plt.plot(xf, np.abs(yf))
    plt.savefig(f'output/{f[4:-4]}fft.png')



    plt.title("Walk")
    plt.xlabel("Time")
    plt.ylabel("Total acceleration")
    plt.figure(figsize=(12,4))


    plt.plot(walk_data["time"], walk_data[col], 'b.')
    plt.plot(walk_data["time"], lowpass, 'r-')

    peaks, _ = signal.find_peaks(lowpass, height=1.5)
    print(peaks)
    plt.plot(walk_data['time'][peaks], lowpass[peaks], "x", color='magenta', markersize=8)
    plt.plot(walk_data['time'], np.zeros_like(lowpass)+2, "--", color="gray")
    plt.savefig(f'output/{f[4:-4]}.png')
