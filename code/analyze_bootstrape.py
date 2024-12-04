import numpy as np


error = np.load('code/bootstrap_errors.npy')
values = np.load('code/bootstrap_lambda_values.npy')
sample_sizes = [500, 1000, 2500, 5000, 10000]
bias = []
mean_error = []
for i in range(len(values)):
    mean_error.append(np.mean(error[i]))
    bias.append(np.mean(values[i]) - 0.3)
    print(np.mean(values[i]))

print(bias)
print(mean_error)