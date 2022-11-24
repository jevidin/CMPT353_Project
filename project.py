import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.nonparametric.smoothers_lowess import lowess
# from pykalman import KalmanFilter

walk_data = sys.argv[1]

cpu_data = pd.read_csv(walk_data, sep=',', parse_dates=['time'], names=['time', 'ax', 'ay', 'az', 'aT'], skiprows=1)
cpu_data['time'] = pd.to_datetime(cpu_data.time.astype(str), format='%H:%M:%S:%f')
print(cpu_data)
cpu_data.info()
plt.title("Walk")
plt.xlabel("Time")
plt.ylabel("Total acceleration")
# plt.locator_params(axis='y', nbins=6)
# plt.locator_params(axis='x', nbins=10)
plt.scatter(cpu_data["time"], cpu_data["aT"])
plt.show()
