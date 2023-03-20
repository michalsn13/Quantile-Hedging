import numpy as np

class Underlying:
    def __init__(self, mu, sigma, r):
        self.mu = mu
        self.sigma = sigma
        self.r = r
    def simulate_P(self, size, T, values_per_year = 365):
        time = np.arange(0,values_per_year * T + 1) / values_per_year
        B = np.random.normal(size=(size, time.shape[0]), loc=0, scale=1)
        W = np.sqrt(time) * B
        sims = np.exp((self.mu - 0.5 * self.sigma**2)*time + self.sigma * W)
        return (B,sims)
    def simulate_Q(self, size, T, values_per_year = 365):
        time = np.arange(0,values_per_year * T + 1) / values_per_year
        B = np.random.normal(size=(size, time.shape[0]), loc=0, scale=1)
        W = np.sqrt(time) * B
        sims = np.exp((self.r - 0.5*self.sigma**2)*time + self.sigma*W)
        return (B, sims)