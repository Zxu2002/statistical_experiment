import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

error_bootstrape = np.load('code/bootstrap_errors.npy')
values_bootstrape = np.load('code/bootstrap_lambda_values.npy')
sample_sizes = [500, 1000, 2500, 5000, 10000]
bias_bootstrape = []
mean_error_bootstrape = []
for i in range(len(values_bootstrape)):
    mean_error_bootstrape.append(np.mean(error_bootstrape[i]))
    bias_bootstrape.append(np.mean(values_bootstrape[i]) - 0.3)
    print(np.mean(values_bootstrape[i]))

#Bootstrape bias and mean error plot
output_dir = Path("report")
fig,(ax0,ax1) = plt.subplots(1,2, figsize=(12,8))
ax0.plot(sample_sizes, bias_bootstrape)
ax0.set_title('Bias')
ax0.set_xlabel("Sample Size")
ax0.set_ylabel("Bias")
ax0.grid(True)
ax1.plot(sample_sizes, mean_error_bootstrape)
ax1.set_title('Mean Error')
ax1.set_xlabel("Sample Size")
ax1.set_ylabel("Error")
plt.suptitle('Bias and Mean Error for different sample sizes for Bootstrapping method on $\lambda$')
output_file = output_dir / "e_bootstrape_plot.png"  
plt.savefig(output_file)   
ax1.grid(True)
plt.show()

error_sweight = np.load('code/sweight_errors.npy')
values_sweight = np.load('code/sweight_lambda_values.npy') 
sample_sizes = [500, 1000, 2500, 5000, 10000]
bias_sweight = []
mean_error_sweight = []
for i in range(len(values_sweight)):
    mean_error_sweight.append(np.mean(error_sweight[i]))
    bias_sweight.append(np.mean(values_sweight[i]) - 0.3)
    print(np.mean(values_sweight[i]))


#sWeight bias and mean error plot
fig,(ax0,ax1) = plt.subplots(1,2,figsize=(12,8))
ax0.plot(sample_sizes, bias_sweight)
ax0.grid(True)
ax0.set_title('Bias')
ax0.set_ylabel("Bias")
ax0.set_xlabel("Sample Size")
ax1.plot(sample_sizes, mean_error_sweight)
ax1.set_title('Mean Error')

ax1.set_ylabel("Error")
ax1.set_xlabel("Sample Size")
plt.suptitle('Bias and Mean Error for different sample sizes for SWeighting method on $\lambda$', fontsize=16)
output_file = output_dir / "f_sweight_plot.png"  
plt.savefig(output_file) 
ax1.grid(True)
plt.show()

#Mixed bias and mean error plot
fig,(ax0,ax1) = plt.subplots(1,2, figsize=(12,8))
ax0.plot(sample_sizes, bias_bootstrape, label='Bootstrape')
ax0.plot(sample_sizes, bias_sweight, label='sWeight')
ax0.set_ylabel("Bias")
ax0.set_xlabel("Sample Size")
ax0.grid(True)
ax0.legend()
ax0.set_title('Bias')
ax1.plot(sample_sizes, mean_error_bootstrape, label='Bootstrape')
ax1.plot(sample_sizes, mean_error_sweight, label='sWeight')
ax1.set_title('Mean Error')
ax1.legend()
ax1.grid(True)
ax1.set_xlabel("Sample Size")
ax1.set_ylabel("Error")
plt.suptitle('Bias and Mean Error for different sample sizes for Bootstrapping and SWeight method on $\lambda$')
output_file = output_dir / "g_comparison_plot.png"  
plt.savefig(output_file)   
plt.show()