import numpy as np
#from total_model import TotalModel
from generate_sample import fitting, generate_sample
from tqdm import tqdm
import os

def bootstrape(toys,mi): 
    '''
    Performs bootstrapping to estimate the error of the parameter lambda.
    '''

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

toys_list = []
for size in sample_sizes:
    N = np.random.poisson(size)
    toys = [generate_sample(N, *mi.values[1:]) for _ in range(250) ]

    toy_x = np.array([toy[0] for toy in toys])
    toy_y = np.array([toy[1] for toy in toys])
    
    # Save toys for this sample size
    np.savez(f'code/toys_size_{size}.npz', 
             toy_x=toy_x, 
             toy_y=toy_y, 
             size=size)
    values, errors = bootstrape(toys, mi)
    val.append(values)
    err.append(errors)
     

np.save('code/bootstrap_lambda_values.npy', np.array(val))
np.save('code/bootstrap_errors.npy', np.array(err))
