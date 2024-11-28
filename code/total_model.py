import numpy as np
from scipy import stats
import scipy 

class TotalModel:
    def __init__(self,mu = 3,sigma = 0.3, beta = 1,m = 1.4,f = 0.6,lamb = 0.3,mu_b = 0,sigma_b = 2.5):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta 
        self.m = m
        self.f = f
        self.lamb = lamb
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        #self.norm_crystal = self.sigma * (self.m * np.exp(-self.beta ** 2/2)/(self.beta * (self.m-1)) + np.sqrt(2 * np.pi) * stats.norm.cdf(self.beta, loc = 0,scale = 1))
        self.norm_crystal = scipy.integrate.quad(lambda x: self.gs_nonorm(x), 0, 5)[0]
        self.hs_norm = 1 - np.exp(-10 * self.lamb)

    def gs_nonorm(self,x):
        z = (x-self.mu)/self.sigma
        if z>-self.beta:
            return np.exp(-z**2/2)
        else:
            frac = self.m/self.beta
            return (frac**(self.m) * np.exp(-self.beta**(2)/2) * (frac - self.beta - z)**(-self.m))
    def gs(self,x):
        return self.gs_nonorm(x)/self.norm_crystal

    def hs(self,y):
        return self.lamb * np.exp(-self.lamb * y)/self.hs_norm
    

    def gb(self,x):
        return stats.uniform.pdf(x,loc = 0,scale = 5)

    def hb(self,y):
        normal_constant = stats.norm.cdf(10, loc = self.mu_b,scale = self.sigma_b) - stats.norm.cdf(0, loc = self.mu_b,scale = self.sigma_b)
        return stats.norm.pdf(y,loc = self.mu_b, scale = self.sigma_b)/normal_constant

    def s(self,y,x):
        return self.gs(x) * self.hs(y)
    
    def b(self,y,x):
        return self.gb(x) * self.hb(y)
    
    def total_func(self,y,x):
        return self.f * self.s(y,x) + (1-self.f) * self.b(y,x)

# total_model = TotalModel()
# result = scipy.integrate.quad(lambda x: total_model.gs(x), 0, 5)
# print(result)
# result = scipy.integrate.quad(lambda y: total_model.hs(y), 0, 10)
# print(result)

# result = scipy.integrate.quad(lambda x: total_model.gb(x), 0, 5)
# print(result)

# result = scipy.integrate.quad(lambda y: total_model.hb(y), 0, 10)
# print(result)

# result = scipy.integrate.dblquad(
#     total_model.s,           
#     0,            
#     5,            
#     lambda x: 0,  
#     lambda x: 10   
# )
# print(result)

# result = scipy.integrate.dblquad(
#     total_model.b,           
#     0,            
#     5,            
#     lambda x: 0,  
#     lambda x: 10   
# )
# print(result)
# result = scipy.integrate.dblquad(
#     total_model.total_func,           
#     0,            
#     5,            
#     lambda x: 0,  
#     lambda x: 10   
# )
# print(result)