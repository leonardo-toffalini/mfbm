from time import perf_counter
from fbm import fbm, wood_chan, wood_chan_np
import numpy as np
import matplotlib.pyplot as plt

data_points = 10
davies_harte_better_times = np.zeros(data_points)
davies_harte_times = np.zeros(data_points)
wood_chan_times = np.zeros(data_points)
wood_chan_np_times = np.zeros(data_points)
hosking_times = np.zeros(data_points)
cholesky_times = np.zeros(data_points)
cholesky_better_times = np.zeros(data_points)
skips = {
    "davies_harte_better": False,
    "daviesharte": False,
    "cholesky": False,
    "hosking": False,
    "wood_chan": False,
    "wood_chan_np": False,
}

Ts = np.linspace(100, 10_000, data_points)

for i, T in enumerate(Ts):
    T = int(T)
    H = 0.7

    if not skips["davies_harte_better"]:
        start = perf_counter()
        process = fbm(length=T, n=T, hurst=H, method="davies_harte_better")
        end = perf_counter()
        davies_harte_better_times[i] = end - start
        if davies_harte_better_times[i] > 20: skips["davies_harte_better"] = True
        
    if not skips["daviesharte"]:
        start = perf_counter()
        process = fbm(length=T, n=T, hurst=H, method="daviesharte")
        end = perf_counter()
        davies_harte_times[i] = end - start
        if davies_harte_times[i] > 20: skips["daviesharte"] = True

    if not skips["cholesky"]:
        start = perf_counter()
        process = fbm(length=T, n=T, hurst=H, method="cholesky")
        end = perf_counter()
        cholesky_times[i] = end - start
        if davies_harte_times[i] > 20: skips["daviesharte"] = True

    if not skips["hosking"]:
        start = perf_counter()
        process = fbm(length=T, n=T, hurst=H, method="hosking")
        end = perf_counter()
        hosking_times[i] = end - start
        if hosking_times[i] > 20: skips["hosking"] = True

    if not skips["wood_chan"]:
        start = perf_counter()
        process = wood_chan(length=T, n=T, H=H)
        end = perf_counter()
        wood_chan_times[i] = end - start
        if wood_chan_times[i] > 20: skips["wood_chan"] = True

    if not skips["wood_chan_np"]:
        start = perf_counter()
        process = wood_chan_np(length=T, n=T, H=H)
        end = perf_counter()
        wood_chan_np_times[i] = end - start
        if wood_chan_np_times[i] > 20: skips["wood_chan_np"] = True

plt.plot(Ts, davies_harte_better_times, label="davies harte np")
plt.plot(Ts, davies_harte_times, label="davies harte")
plt.plot(Ts, hosking_times, label="hosking")
plt.plot(Ts, cholesky_times, label="cholesky")
plt.plot(Ts, wood_chan_times, label="wood chan")
plt.plot(Ts, wood_chan_np_times, label="wood chan np")
plt.legend()
plt.show()

