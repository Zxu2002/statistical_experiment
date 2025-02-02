import numpy as np
from scipy import stats
import scipy 
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

class TotalModel:
    '''
    This class defines the total model for the signal and background distributions.

    Parameters:
    mu: float
        Mean of the signal distribution, x-coordinate
    sigma: float
        Standard deviation of the signal distribution, x-coordinate
    beta: float 
        Parameter for the signal distribution, x-coordinate
    m: float    
        Parameter for the signal distribution, x-coordinate
    f: float
        Fraction of signal events, between 0 and 1
    lamb: float
        Parameter for the signal distribution, y-coordinate
    mu_b: float
        Mean of the background distribution, y-coordinate
    sigma_b: float
        Standard deviation of the background distribution, y-coordinate
    '''
    def __init__(self,mu = 3,sigma = 0.3, beta = 1,m = 1.4,f = 0.6,lamb = 0.3,mu_b = 0,sigma_b = 2.5):
        '''
        This function initializes the TotalModel class.
        '''
        self.mu = mu
        self.sigma = sigma
        self.beta = beta 
        self.m = m
        self.f = f
        self.lamb = lamb
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        #self.norm_crystal = self.sigma * (self.m * np.exp(-self.beta ** 2/2)/(self.beta * (self.m-1)) + np.sqrt(2 * np.pi) * stats.norm.cdf(self.beta, loc = 0,scale = 1))
        self.norm_crystal = scipy.integrate.quad(lambda x: self.gs_nonorm(x), 0, 5,epsabs=1e-6, epsrel=1e-6)[0]
        self.hs_norm = 1 - np.exp(-10 * self.lamb)

    def gs_nonorm(self,x):
        '''
        This function defines the signal distribution on x-coordinate before the normalization process.
        '''
        z = np.array((x-self.mu)/self.sigma)
        result = np.zeros_like(z, dtype=float)
        
        mask = z > -self.beta
        result[mask] = np.exp(-z[mask]**2/2)
        
        frac = self.m/self.beta
        mask_tail = ~mask
        valid_tail = mask_tail & (frac - self.beta - z > 0)
        #Added safety check to avoid power of negative numbers
        if np.any(valid_tail):
            result[valid_tail] = (frac**self.m * np.exp(-self.beta**2/2) * 
                           (frac - self.beta - z[valid_tail])**(-self.m))
        return result

    def gs(self,x):
        '''
        This function defines the signal distribution after the normalization process.
        '''
        return self.gs_nonorm(x)/self.norm_crystal

    def hs(self,y):
        '''
        This function defines the signal distribution on y-coordinate.
        '''
        return self.lamb * np.exp(-self.lamb * y)/self.hs_norm
    

    def gb(self,x):
        '''
        This function defines the background distribution on x-coordinate.
        '''
        return stats.uniform.pdf(x,loc = 0,scale = 5)

    def hb(self,y):
        '''
        This function defines the background distribution on y-coordinate.
        '''
        normal_constant = stats.norm.cdf(10, loc = self.mu_b,scale = self.sigma_b) - stats.norm.cdf(0, loc = self.mu_b,scale = self.sigma_b)
        return stats.norm.pdf(y,loc = self.mu_b, scale = self.sigma_b)/normal_constant

    def s(self,y,x):
        '''
        This function defines the signal distribution on both x and y coordinates.
        '''
        return self.gs(x) * self.hs(y)
    
    def b(self,y,x):
        '''
        This function defines the background distribution on both x and y coordinates.
        '''
        return self.gb(x) * self.hb(y)
    
    def total_func(self,y,x):
        '''
        This function defines the total distribution on both x and y coordinates.
        '''
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
if __name__ == "__main__":
    model = TotalModel()
    x_vals = np.linspace(0, 5, 500)
    y_vals = np.linspace(0, 10, 500)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = model.total_func(Y, X)

    # Plot X distributions
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    axes[0].plot(x_vals, model.gs(x_vals), label='Signal')
    axes[0].plot(x_vals, model.gb(x_vals), label='Background')
    axes[0].plot(x_vals, model.f * model.gs(x_vals) + (1-model.f) * model.gb(x_vals), label='Total')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('PDF')
    axes[0].legend()
    axes[0].grid(alpha=0.5)
    axes[0].set_title('1D Projection in X')


    # Plot Y distributions
    axes[1].plot(y_vals, model.hs(y_vals), label='Signal')
    axes[1].plot(y_vals, model.hb(y_vals), label='Background')
    axes[1].plot(y_vals, model.f * model.hs(y_vals) + (1-model.f) * model.hb(y_vals), label='Total')
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('PDF')
    axes[1].legend()
    axes[1].grid(alpha=0.5)
    axes[1].set_title('1D Projection in Y')

    #fig.suptitle('One Dimensional Projections', fontsize=16)
    output_dir = Path("report")  
    output_file = output_dir / "q3_1d_projection.png"  

    plt.savefig(output_file) 
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()
    # Plot 2D joint distribution
    plt.figure()
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Joint PDF')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Joint Probability Density')
    output_file = output_dir / "q3_2d_plot.png"  
    plt.savefig(output_file) 
    plt.show()