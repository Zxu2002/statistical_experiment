import numpy as np
from scipy import stats
from scipy import optimize
import timeit
#sys.path.append(str(Path(__file__).resolve().parent.parent))
from total_model import TotalModel
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL

#Given there's no analytical ppf use accept-reject method
def generate_sample(num_sample):
    x_grid = np.linspace(0, 5, 100)
    y_grid = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    model = TotalModel() 
    Z = model.total_func(Y, X)
    max_density = np.max(Z) * 1.1 

    x_samples, y_samples= [], []

    while len(x_samples) < num_sample:
        batch_size = (num_sample - len(x_samples)) * 4
        x_candi = np.random.uniform(0, 5, size=batch_size)
        y_candi = np.random.uniform(0, 10, size=batch_size)
        pdf_val = model.total_func(y_candi,x_candi)
        u = np.random.uniform(0, max_density, size=batch_size)
        accepted = u < pdf_val
        x_samples.extend(x_candi[accepted])
        y_samples.extend(y_candi[accepted])

    x_samples = np.array(x_samples[:num_sample])
    y_samples = np.array(y_samples[:num_sample])
    return x_samples, y_samples

if __name__ == "__main__":
    model = TotalModel()
    x, y = generate_sample(10000)
    z = np.random.normal(size=100000)
    nll = ExtendedUnbinnedNLL((y,x), model.total_func)
    mi = Minuit(nll, N=1000, mu=0, sg=1)
    mi.migrad()
    mi.hesse()