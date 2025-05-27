from time import perf_counter
from fbm import fbm

T = 10_000_000

start = perf_counter()
process = fbm(length=T, n=T, hurst=0.1, method="davies_harte_better")
end = perf_counter()

print("davies_harte_better", end - start)

start = perf_counter()
process = fbm(length=T, n=T, hurst=0.1, method="daviesharte")
end = perf_counter()

print("davies_harte", end - start)
