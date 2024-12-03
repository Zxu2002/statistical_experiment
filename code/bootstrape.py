import numpy as np
from total_model import TotalModel
from generate_sample import fitting, fit_func, generate_sample
from tqdm import tqdm

def bootstrape(sample_size,mi,n = 250): 
    toys = [generate_sample(np.random.poisson(sample_size), *mi.values[1:]) for _ in range(n) ]
    values = []
    errors = []
    for toy in tqdm(toys):
        #print(toy) 
        mi_t = fitting(toy[0],toy[1])
        values.append( list(mi_t.values) ) 
        errors.append( list(mi_t.errors) )
    return np.array(values), np.array(errors)


x, y = generate_sample(100000)
mi = fitting(x, y)
print(*mi.values[1:])
sample_sizes = [500, 1000, 2500, 5000, 10000]
val = []
err = []
for i in sample_sizes:
    values, errors = bootstrape(i,mi)
    val.append(values)
    err.append(errors)