import numpy as np
# from scipy import stats
# from scipy import optimize
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

def fit_func(data, N, mu, sigma, beta, m, f, lamb, mu_b, sigma_b):
    x, y = data
    model = TotalModel(mu, sigma, beta, m, f, lamb, mu_b, sigma_b)
    return N, N * model.total_func(y, x)

def fitting(x,y):
    nll = ExtendedUnbinnedNLL([x,y], fit_func)
    print("fitting")
    mi = Minuit(nll, N = 100000, mu = 0,sigma = 1, beta = 1,m = 1,f = 0.5,lamb = 0.1,mu_b = 0.5,sigma_b = 1)
    mi.migrad()
    mi.hesse()
    return mi 

def run_timeit(num_events=100000, num_trials=100):
    t1 = timeit.timeit(lambda: np.random.normal(size=num_events), number=num_trials)
    t2 = timeit.timeit(lambda: generate_sample(num_events), number=num_trials)
    x, y = generate_sample(num_events)
    t3 = timeit.timeit(lambda: fitting(x, y), number=num_trials)
    
    return t1/num_trials, t2/num_trials, t3/num_trials

if __name__ == "__main__":
    model = TotalModel()

    x, y = generate_sample(100000)
    mi = fitting(x, y)
    #z = np.random.normal(size=100000)
    t_norm, t_gen_samp, t_fit_samp = run_timeit()
    print(f"time, benchmark(normal distribution generate): {t_norm}")
    print(f"time. generate joint distribution sample: {t_gen_samp}")
    print(f"time. fit joint distribution sample: {t_fit_samp}")
# time, benchmark(normal distribution generate): 0.002388919999999999
# time. generate joint distribution sample: 0.12924637
# time. fit joint distribution sample: 9.89350620375
    for param in mi.parameters:
        value = mi.values[param]
        error = mi.errors[param]
        print(f"{param}, value: {value}, error: {error}")
# N, value: 100000.11566607063, error: 316.2278705715917
# mu, value: 2.995978939197134, error: 0.002547711444145466
# sigma, value: 0.2980989755225555, error: 0.0024455735633328955
# beta, value: 1.0144100338780004, error: 0.02188330129199616
# m, value: 1.3503588518086418, error: 0.05831105257700519
# f, value: 0.5951146318907781, error: 0.0035400929404916254
# lamb, value: 0.2973744739614957, error: 0.002062927464454676
# mu_b, value: 0.12313059108177564, error: 0.07245203573491325
# sigma_b, value: 2.442622761880231, error: 0.034776180638041686