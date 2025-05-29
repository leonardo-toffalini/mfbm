from time import perf_counter
from fbm import fbm, wood_chan, wood_chan_np
import numpy as np
import matplotlib.pyplot as plt

data_points = 100
davies_harte_better_times = np.zeros(data_points)
davies_harte_times = np.zeros(data_points)
wood_chan_times = np.zeros(data_points)
wood_chan_np_times = np.zeros(data_points)

Ts = np.linspace(100, 1_000_000, data_points)

for i, T in enumerate(Ts):
    T = int(T)
    H = 0.7

    start = perf_counter()
    process = fbm(length=T, n=T, hurst=H, method="davies_harte_better")
    end = perf_counter()
    davies_harte_better_times[i] = end - start

    start = perf_counter()
    process = fbm(length=T, n=T, hurst=H, method="daviesharte")
    end = perf_counter()

    davies_harte_times[i] = end - start


    start = perf_counter()
    process = wood_chan(length=T, n=T, H=H)
    end = perf_counter()

    wood_chan_times[i] = end - start

    start = perf_counter()
    process = wood_chan_np(length=T, n=T, H=H)
    end = perf_counter()

    wood_chan_np_times[i] = end - start

plt.plot(Ts, davies_harte_better_times, label="davies harte np")
plt.plot(Ts, davies_harte_times, label="davies harte")
plt.plot(Ts, wood_chan_times, label="wood chan")
plt.plot(Ts, wood_chan_np_times, label="wood chan np")
plt.legend()
plt.show()

