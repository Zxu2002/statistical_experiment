import numpy as np
from total_model import TotalModel
from generate_sample import fitting, fit_func, generate_sample


x, y = generate_sample(100000)
mi = fitting(x, y)

sample_sizes = [500, 1000, 2500, 5000, 10000]