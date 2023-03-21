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