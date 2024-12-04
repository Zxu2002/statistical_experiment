import numpy as np
from generate_sample import fitting, generate_sample
from tqdm import tqdm
from sweights import Cows 
from total_model import TotalModel
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL

def x_model(x, N, mu, sigma, beta, m, f):
    '''
    The x model. 
    '''
    lamb = 0.3
    mu_b = 0
    sigma_b = 2.5
    model = TotalModel(mu, sigma, beta, m, f, lamb, mu_b, sigma_b)
    target = f * model.gs(x) + (1 - f) * model.gb(x)
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

# mu = 3,sigma = 0.3, beta = 1,m = 1.4,f = 0.6,lamb = 0.3,mu_b = 0,sigma_b = 2.5

if __name__ == "__main__":
    x,y = generate_sample(250)
    weights = get_sweights(x)