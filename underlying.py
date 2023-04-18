import numpy as np
import pandas as pd

class Underlying:
    def __init__(self, mu, sigma, r, values_per_year = 365):
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.values_per_year = values_per_year
    def simulate_P(self, size, T):
        price_moments = np.arange(0,self.values_per_year * T + 1)
        Sigma=1/self.values_per_year*np.minimum(np.tile(price_moments,(len(price_moments),1)),np.tile(price_moments.reshape(-1,1),(1,len(price_moments))))
        B = pd.DataFrame(np.random.multivariate_normal(size=size, mean= np.zeros(len(price_moments)), cov = Sigma))
        sims = np.exp((self.mu - 0.5 * self.sigma**2) * price_moments / self.values_per_year + self.sigma * B)
        return (B,sims)
    def simulate_Q(self, size, T):
        price_moments = np.arange(0,self.values_per_year * T + 1)
        Sigma=1/self.values_per_year*np.minimum(np.tile(price_moments,(len(price_moments),1)),np.tile(price_moments.reshape(-1,1),(1,len(price_moments))))
        B = pd.DataFrame(np.random.multivariate_normal(size=size, mean= np.zeros(len(price_moments)), cov = Sigma))
        sims = np.exp((self.r - 0.5 * self.sigma**2) * price_moments / self.values_per_year + self.sigma * B)
        return (B,sims)                
    
class NonTradedUnderlying(Underlying):
    def __init__(self, mu, sigma, underlying_t, rho):
        self.underlying_t = underlying_t
        self.rho = rho
        Underlying.__init__(self, mu, sigma, underlying_t.r, underlying_t.values_per_year)
    def simulate_together_P(self, size, T):
        price_moments = np.arange(0,self.values_per_year * T + 1)
        Sigma=1/self.values_per_year*np.minimum(np.tile(price_moments,(len(price_moments),1)),np.tile(price_moments.reshape(-1,1),(1,len(price_moments))))
        B_t = pd.DataFrame(np.random.multivariate_normal(size=size, mean= np.zeros(len(price_moments)), cov = Sigma))
        B_nt = self.rho * B_t + np.sqrt(1 - self.rho**2) * pd.DataFrame(np.random.multivariate_normal(size=size, mean= np.zeros(len(price_moments)), cov = Sigma))
        sims_t = np.exp((self.underlying_t.mu - 0.5 * self.underlying_t.sigma**2) * price_moments / self.values_per_year + self.underlying_t.sigma * B_t)
        sims_nt = np.exp((self.mu - 0.5 * self.sigma**2) * price_moments / self.values_per_year + self.sigma * B_nt)
        return [(B_t,sims_t), (B_nt, sims_nt)]
    def simulate_together_Q(self, size, T):
        price_moments = np.arange(0,self.values_per_year * T + 1)
        Sigma=1/self.values_per_year*np.minimum(np.tile(price_moments,(len(price_moments),1)),np.tile(price_moments.reshape(-1,1),(1,len(price_moments))))
        B_t = pd.DataFrame(np.random.multivariate_normal(size=size, mean= np.zeros(len(price_moments)), cov = Sigma))
        B_nt = self.rho * B_t + np.sqrt(1 - self.rho**2) * pd.DataFrame(np.random.multivariate_normal(size=size, mean= np.zeros(len(price_moments)), cov = Sigma))
        sims_t = np.exp((self.r - 0.5 * self.underlying_t.sigma**2) * price_moments / self.values_per_year + self.underlying_t.sigma * B_t)
        sims_nt = np.exp((self.r - 0.5 * self.sigma**2) * price_moments / self.values_per_year + self.sigma * B_nt)
        return [(B_t,sims_t), (B_nt, sims_nt)]


        
    