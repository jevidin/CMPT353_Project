import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.nonparametric.smoothers_lowess import lowess
import glob
from scipy import signal
# from pykalman import KalmanFilter

for f in glob.glob(sys.argv[1]):
    walk_data = pd.read_csv(f, sep=',', parse_dates=['time'], names=['time', 'ax', 'ay', 'az', 'aT'], skiprows=1)
    walk_data['time'] = pd.to_datetime(walk_data.time.astype(str), format='%H:%M:%S:%f')
    b, a = signal.butter(3, 0.08, btype='lowpass', analog=False)
    lowpass = signal.filtfilt(b, a, walk_data["aT"])
    plt.title("Walk")
    plt.xlabel("Time")
    plt.ylabel("Total acceleration")
    plt.figure(figsize=(12,4))
    plt.plot(walk_data["time"], walk_data["aT"], 'b.')
    plt.plot(walk_data["time"], lowpass, 'r-')
    plt.savefig(f'output/{f[4:-4]}.png')
