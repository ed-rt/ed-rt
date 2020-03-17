from math import sqrt
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
print(num_cores)

l = Parallel(n_jobs=num_cores)(delayed(sqrt)(i**2) for i in range(10))

print(l)