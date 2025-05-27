from time import perf_counter
from fbm import fbm

T = 100_000
# start = perf_counter()
# process = fbm(length=T, n=T, hurst=0.1, method="cholesky")
# end = perf_counter()
#
# print("cholesky", end - start)

start = perf_counter()
process = fbm(length=T, n=T, hurst=0.1, method="cholesky_better")
end = perf_counter()

print("cholesky_better", end - start)

