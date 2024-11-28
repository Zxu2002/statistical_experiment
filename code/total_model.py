import numpy as np
from scipy import stats

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
        self.norm_crystal = self.sigma * (self.m * np.exp(-self.beta ** 2/2)/(self.beta * (self.m-1)) + np.sqrt(2 * np.pi) * stats.norm.cdf(self.beta, loc = 0,scale = 1))

    def gs(self,x):
        z = (x-self.mu)/self.sigma
        
        if z>-self.beta:
            return self.norm_crystal * np.exp(-z^2/2)
        else:
            frac = self.m/self.beta
            return self.norm_crystal * (frac^(self.m) * np.exp(-self.beta^(2)/2) * (frac - self.beta - z)^(-self.m) )


    def hs(self,y):
        return self.lamb * np.exp(-self.lamb * y)

    def gb(self,x):
        return stats.uniform.pdf(x,loc = 0,scale = 5)

    def hb(self,y):
        normal_constant = stats.norm.cdf(10, loc = self.mu_b,scale = self.sigma_b) - stats.norm.cdf(0, loc = self.mu_b,scale = self.sigma_b)
        return stats.norm(y,loc = self.mu_b,scale = self.sigma_b)/normal_constant

    def s(self,x,y):
        return self.gs(x) * self.hs(y)
    
    def b(self,x,y):
        return self.gb(x) * self.hb(y)
    
    def total_func(self,x,y):
        return self.f * self.s(x,y) + (1-self.f) * self.b(x,y)

        