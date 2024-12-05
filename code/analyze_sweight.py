import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

error = np.load('code/sweight_errors.npy')
values = np.load('code/sweight_lambda_values.npy') 
sample_sizes = [500, 1000, 2500, 5000, 10000]
bias = []
mean_error = []
for i in range(len(values)):
    mean_error.append(np.mean(error[i]))
    bias.append(np.mean(values[i]) - 0.3)
    print(np.mean(values[i]))

print(bias)
print(mean_error)
output_dir = Path("report")  

fig,(ax0,ax1) = plt.subplots(2,1)
ax0.plot(sample_sizes, bias)
ax0.set_title('Bias')
ax1.plot(sample_sizes, mean_error)
ax1.set_title('Mean Error')
plt.suptitle('Bias and Mean Error for different sample sizes for SWeighting method on lambda', fontsize=16)
output_file = output_dir / "f_sweight_plot.png"  
plt.savefig(output_file) 
plt.show()