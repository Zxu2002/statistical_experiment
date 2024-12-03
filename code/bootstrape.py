import numpy as np

from generate_sample import generate_sample, fitting, fit_func
x, y = generate_sample(100000)
mi = fitting(x, y)