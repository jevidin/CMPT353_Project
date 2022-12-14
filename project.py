import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import signal, stats
from scipy.fft import rfftfreq, fft
import datetime
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

os.makedirs("output/", exist_ok=True)
col = 'aT'

# all files in data
for f in glob.glob(sys.argv[1]):
    walk_data = pd.read_csv(f, sep=',', parse_dates=['time'], names=['time', 'ax', 'ay', 'az', 'aT'], skiprows=1)
    walk_data['time'] = pd.to_datetime(walk_data.time.astype(str), format='%H:%M:%S:%f')
    last_time = walk_data.iloc[-1]['time'] - datetime.timedelta(seconds=5)
    walk_data = walk_data[(walk_data['time'] <= last_time)]

    #cut off extra unnecessary data, remove data before the first step
    tmp = walk_data[(walk_data[col]>=1.5)]
    first_time = tmp.iloc[0]['time']
    #deletes rows before that time
    walk_data = walk_data[(walk_data['time'])>=first_time]
    walk_data = walk_data.reset_index()

    #Butterworth filter
    b, a = signal.butter(3, 0.02, btype='lowpass', analog=False)
    lowpass = signal.filtfilt(b, a, walk_data[col])

    #find peaks above 1.5
    peak_heights = 1.5
    peaks, _ = signal.find_peaks(lowpass, height=peak_heights)

    # fourier transform
    # https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
    X = fft(lowpass)
    N = len(X)
    n_oneside = N//2
    f_oneside = rfftfreq(N, 1/205)[:n_oneside]

    # period
    t_h = 1/ f_oneside    

    # plot frequency graph
    plt.figure(figsize=(12,6))
    plt.plot(t_h, np.abs(X[:n_oneside])/n_oneside)
    plt.title(f"Frequencies for {f[5:-4]}")
    plt.xlabel('Period (seconds)')
    plt.savefig(f'output/{f[5:-4]}fft.png')

    # plot filtered data
    plt.figure(figsize=(12,4))
    plt.title(f"Walk data for {f[5:-4]}")
    plt.xlabel("Time")
    plt.ylabel("Total acceleration")
    plt.plot(walk_data["time"], walk_data[col], 'b.', markersize=1)
    plt.plot(walk_data["time"], lowpass, 'r-')
    plt.plot(walk_data['time'][peaks], lowpass[peaks], "x", color='magenta', markersize=8)
    plt.plot(walk_data['time'], np.zeros_like(lowpass)+peak_heights, "--", color="gray")
    plt.savefig(f'output/{f[5:-4]}.png')

    # Peak analysis
    peak_vals = lowpass[peaks]
    odd = peak_vals[1::2]
    even = peak_vals[0::2]
    if len(peak_vals > 10):
        peak_vals = peak_vals[:10]
    # print(len(peak_vals))
    mann_u_p = stats.mannwhitneyu(odd, even).pvalue
    print(f"MANN U p-val for {f[5:-4]} : {mann_u_p}")
