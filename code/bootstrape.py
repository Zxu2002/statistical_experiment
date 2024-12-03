import numpy as np
from scipy import stats
from scipy import optimize
import timeit
from generate_sample import generate_sample
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL