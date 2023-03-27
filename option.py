import numpy as np
import pandas as pd
from scipy.stats import norm

class Option:
    def __init__(self, underlying, payoff_func, T, MC_setup_max = 10000):
        self.underlying = underlying
        self.payoff_func = payoff_func
        self.T = T
        self.MC_setup_max = MC_setup_max
        self.MC_setup = self.underlying.simulate_Q(MC_setup_max, self.T)
    def reset_MC_setup(self):
        self.MC_setup = self.underlying.simulate_Q(self.MC_setup_max, self.T)
    def get_MC_price(self, X0_rel, t = 0, n_sims = 10000):
        if t > self.T:
            raise Exception(f'{t}> {self.T}: Pricing moment cannot exceed option expirancy moment T={self.T}')
        if self.MC_setup_max < n_sims:
            self.MC_setup_max = n_sims
            self.reset_MC_setup()
        discount = np.exp(-self.underlying.r * (self.T - t)) 
        B_full, sims_full = self.MC_setup
        final_index = round(self.underlying.values_per_year * (self.T - t) + 1)
        B, sims = B_full.iloc[:n_sims,:final_index], sims_full.iloc[:n_sims,:final_index]
        payoffs = self.payoff_func(X0_rel * sims)
        payoffs_mean = payoffs.mean()
        MC_B = B.iloc[:,-1].mean()
        rho = np.sum((payoffs-payoffs_mean)*(B.iloc[:,-1]-0))/(payoffs.shape[0]-1)
        price = payoffs_mean - rho/self.T * (MC_B - 0)
        return discount * price
    def get_MC_delta(self, X0_rel, t = 0, dX = 5, n_sims = 10000):
        price_minus = self.get_MC_price(X0_rel-dX, t, n_sims)
        price_plus = self.get_MC_price(X0_rel+dX, t, n_sims)
        delta = (price_plus - price_minus)/(2*dX)
        return delta
    
class Vanilla(Option):
    def __init__(self, underlying, K, T, call):
        self.K = K
        self.call = call
        if self.call:
            def payoff_func(X):
                return np.maximum(X.iloc[:,-1] - K, 0)
        else:
            def payoff_func(X):
                return np.maximum(K - X.iloc[:,-1], 0)
        Option.__init__(self, underlying, payoff_func, T)
    def get_price(self, X0_rel, t = 0, qh_boundary = None):
        if t > self.T:
            raise Exception(f'{t}> {self.T}: Pricing moment cannot exceed option expirancy moment T={self.T}')
        if t == self.T:
            if call:
                return (X0_rel >= np.max(0,qh_boundary)) * np.max(X0_rel - self.K, 0)
            else:
                return (X0_rel >= np.max(0,qh_boundary)) * np.max(self.K - X0_rel, 0)
        d1 = (np.log(X0_rel/self.K) + (self.underlying.r + 0.5 * self.underlying.sigma**2)*(self.T - t))/(self.underlying.sigma * np.sqrt(self.T - t))
        d2 = d1 - self.underlying.sigma * np.sqrt(self.T - t)
        call_K = X0_rel * norm.cdf(d1) - self.K * np.exp(-self.underlying.r * (self.T - t)) * norm.cdf(d2)
        put_K = self.K * np.exp(-self.underlying.r * (self.T - t)) * norm.cdf(-d2) - X0_rel * norm.cdf(-d1)
        if qh_boundary:
            d1_c = (np.log(X0_rel/qh_boundary) + (self.underlying.r + 0.5 * self.underlying.sigma**2)*(self.T - t))/(self.underlying.sigma * np.sqrt(self.T - t))
            d2_c = d1_c - self.underlying.sigma * np.sqrt(self.T - t)
            call_c = X0_rel * norm.cdf(d1_c) - qh_boundary * np.exp(-self.underlying.r * (self.T - t)) * norm.cdf(d2_c)
            binary_call_c = np.exp(-self.underlying.r * (self.T - t)) * norm.cdf(d2_c)
            if self.call:
                return call_K - call_c - (qh_boundary - self.K) * binary_call_c
            else:
                return put_K + call_c - call_K - (self.K - qh_boundary) * binary_call_c  
        else:
            if self.call:
                return call_K
            else:
                return put_K
    def get_delta(self, X0_rel, t = 0, qh_boundary = None):
        if t == self.T:
            return 0
        d1 = (np.log(X0_rel/self.K) + (self.underlying.r + 0.5 * self.underlying.sigma**2)*(self.T - t))/(self.underlying.sigma * np.sqrt(self.T - t))
        d2 = d1 - self.underlying.sigma * np.sqrt(self.T - t)
        call_K = norm.cdf(d1)
        put_K = norm.cdf(d1) - 1
        if qh_boundary:
            d1_c = (np.log(X0_rel/qh_boundary) + (self.underlying.r + 0.5 * self.underlying.sigma**2)*(self.T - t))/(self.underlying.sigma * np.sqrt(self.T - t))
            d2_c = d1_c - self.underlying.sigma * np.sqrt(self.T - t)
            call_c = norm.cdf(d1_c)
            binary_call_c = np.exp(-self.underlying.r * (self.T - t))*norm.pdf(d2_c)/(self.underlying.sigma * X0_rel * np.sqrt(self.T - t))
            if self.call:
                return call_K - call_c - (qh_boundary - self.K) * binary_call_c
            else:
                return put_K + call_c - call_K - (self.K - qh_boundary) * binary_call_c  
        else:
            if self.call:
                return call_K
            else:
                return put_K
        
            
            
        
