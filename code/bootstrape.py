import numpy as np
#from total_model import TotalModel
from generate_sample import fitting, generate_sample
from tqdm import tqdm

def bootstrape(sample_size,mi,n = 250): 
    N = np.random.poisson(sample_size)
    toys = [generate_sample(N, *mi.values[1:]) for _ in range(n) ]
    values = []
    errors = []
    for toy in tqdm(toys):
        #print(toy) 
        mi_t = fitting(toy[0],toy[1])
        lambda_hat = mi_t.values["lamb"]
        lambda_err = mi_t.errors["lamb"]
        values.append(lambda_hat) 
        errors.append(lambda_err)
    return np.array(values), np.array(errors)


x, y = generate_sample(100000)
mi = fitting(x, y)
print(*mi.values[1:])
sample_sizes = [500, 1000, 2500, 5000, 10000]
val = []
err = []
for size in sample_sizes:
    values, errors = bootstrape(size, mi)
    
    val.append(values)
    err.append(errors)

np.save('code/bootstrap_lambda_values.npy', np.array(val))
np.save('code/bootstrap_errors.npy', np.array(err))
