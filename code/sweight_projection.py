import numpy as np
from generate_sample import fitting, generate_sample
from tqdm import tqdm
from sweights import Cows 
from total_model import TotalModel
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from sweights.util import plot_binned, make_weighted_negative_log_likelihood


def x_model(x, N, mu, sigma, beta, m, f):
    '''
    The x model. 
    '''

    model = TotalModel(mu = mu, sigma=sigma, beta=beta, m=m, f=f)
    target = f * model.gs(x) + (1 - f) * model.gb(x)
    # print(f"gs shape: {np.shape(model.gs(x))}")
    # print(f"gb shape: {np.shape(model.gs(x))}")
    return N, N * target



def fitting_x(x):
    '''
    Fit the x model to the data in x coordinate.
    '''
    nll = ExtendedUnbinnedNLL(x, x_model)
    print("fitting")
    mi = Minuit(nll, N = len(x), mu = 0,sigma = 1, beta = 1,m = 1,f = 0.5)
    mi.limits['N'] = (0, None)
    mi.limits['mu'] = (-3, 3)
    mi.limits['sigma'] = (0, 1)
    mi.limits['beta'] = (0, 3)
    mi.limits['m'] = (1, 3)
    mi.limits['f'] = (0, 1)


    mi.migrad()
    mi.hesse()
    return mi 

def get_sweights(x, lamb = 0.3, mu_b = 0, sigma_b = 2.5):
    '''
    Get the sWeights from the x model.
    '''
    mi = fitting_x(x)
    model = TotalModel(*mi.values['mu','sigma', 'beta','m','f'], lamb, mu_b, sigma_b)

    sw = Cows(x, model.gs, model.gb)
    return sw(x)

def estimate_y(x, y):
    '''
    Estimate the y model using the sWeights.
    '''
    
    # print(f"x shape: {np.shape(x)}")
    # print(f"y shape: {np.shape(y)}")

    def y_model(y, lamb, mu_b, sigma_b):
        #The y model.
        model = TotalModel(*mi.values['mu','sigma', 'beta','m','f'], lamb, mu_b, sigma_b)
        target = model.f * model.hs(y) + (1 - model.f) * model.hb(y)
        return target
    
    mi = fitting_x(x)
    sw = get_sweights(x)
    weighted_nll = make_weighted_negative_log_likelihood(y, sw, y_model)
    ymi = Minuit(weighted_nll, lamb = 0.1,mu_b = 0.5,sigma_b = 1)
    
    ymi.limits['lamb'] = (0, 3)
    ymi.limits['mu_b'] = (-3, 3)
    ymi.limits['sigma_b'] = (0, 3)
    ymi.migrad()
    ymi.hesse()
    return ymi 

if __name__ == "__main__":
    sample_sizes = [500, 1000, 2500, 5000, 10000]
    val = []
    err = []
    for size in sample_sizes:
        val_temp = []
        err_temp = []
        N = np.random.poisson(size)
        for _ in  tqdm(range(250)):
            x,y = generate_sample(N)
            ymi = estimate_y(x, y)
            val_temp.append(ymi.values['lamb'])
            err_temp.append(ymi.errors['lamb'])

        val.append(val_temp)
        err.append(val_temp)
    np.save('code/sweight_lambda_values.npy', np.array(val))
    np.save('code/sweight_errors.npy', np.array(err))
