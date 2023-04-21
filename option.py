import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

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
    def get_MC_delta(self, X0_rel, t = 0, dX = 10, n_sims = 10000):
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
                return (X.iloc[:,-1] <= c1) * (X.iloc[:,-1] >= c2) * np.max(X0_rel - self.K, 0)
            else:
                return (X.iloc[:,-1] <= c1) * (X.iloc[:,-1] >= c2) * np.max(self.K - X0_rel, 0)
        
        d1 = (np.log(X0_rel/self.K) + (self.underlying.r + 0.5 * self.underlying.sigma**2)*(self.T - t))/(self.underlying.sigma * np.sqrt(self.T - t))
        d2 = d1 - self.underlying.sigma * np.sqrt(self.T - t)
        call_K = X0_rel * norm.cdf(d1) - self.K * np.exp(-self.underlying.r * (self.T - t)) * norm.cdf(d2)
        put_K = self.K * np.exp(-self.underlying.r * (self.T - t)) * norm.cdf(-d2) - X0_rel * norm.cdf(-d1)
        if qh_boundary:
            call_cs = []
            put_cs = []
            binary_call_cs = []
            binary_put_cs = []
            for c in qh_boundary:
                if not c:
                    call_cs.append(0)
                    put_cs.append(0)
                    binary_call_cs.append(0)
                    binary_put_cs.append(0)
                else:
                    d1_c = (np.log(X0_rel/c) + (self.underlying.r + 0.5 * self.underlying.sigma**2)*(self.T - t))/(self.underlying.sigma * np.sqrt(self.T - t))
                    d2_c = d1_c - self.underlying.sigma * np.sqrt(self.T - t)
                    call_cs.append(X0_rel * norm.cdf(d1_c) - c * np.exp(-self.underlying.r * (self.T - t)) * norm.cdf(d2_c))
                    put_cs.append(c * np.exp(-self.underlying.r * (self.T - t)) * norm.cdf(-d2_c) - X0_rel * norm.cdf(-d1_c))
                    binary_call_cs.append(np.exp(-self.underlying.r * (self.T - t)) * norm.cdf(d2_c))
                    binary_put_cs.append(np.exp(-self.underlying.r * (self.T - t)) * (1 - norm.cdf(d2_c)))
            if self.call:
                return call_K - call_cs[0] - (qh_boundary[0] - self.K) * binary_call_cs[0] + call_cs[1] + (qh_boundary[1] - self.K) * binary_call_cs[1]
            else:
                return put_K - put_cs[1] - (self.K - qh_boundary[1]) * binary_put_cs[1] + put_cs[0] + (self.K  - qh_boundary[0]) * binary_put_cs[0]
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
            call_cs = []
            put_cs = []
            binary_call_cs = []
            binary_put_cs = []
            for c in qh_boundary:
                if not c:
                    call_cs.append(0)
                    put_cs.append(0)
                    binary_call_cs.append(0)
                    binary_put_cs.append(0)
                else:
                    d1_c = (np.log(X0_rel/c) + (self.underlying.r + 0.5 * self.underlying.sigma**2)*(self.T - t))/(self.underlying.sigma * np.sqrt(self.T - t)) 
                    d2_c = d1_c - self.underlying.sigma * np.sqrt(self.T - t)
                    call_cs.append(norm.cdf(d1_c))
                    put_cs.append(norm.cdf(d1_c) - 1)
                    binary_call_cs.append(np.exp(-self.underlying.r * (self.T - t))*norm.pdf(d2_c)/(self.underlying.sigma * X0_rel * np.sqrt(self.T - t)))
                    binary_put_cs.append(- np.exp(-self.underlying.r * (self.T - t))*norm.pdf(d2_c)/(self.underlying.sigma * X0_rel * np.sqrt(self.T - t)))
            if self.call:
                return call_K - call_cs[0] - (qh_boundary[0] - self.K) * binary_call_cs[0] + call_cs[1] + (qh_boundary[1] - self.K) * binary_call_cs[1]
            else:
                return put_K - put_cs[1] - (self.K - qh_boundary[1]) * binary_put_cs[1] + put_cs[0] + (self.K  - qh_boundary[0]) * binary_put_cs[0]
        else:
            if self.call:
                return call_K
            else:
                return put_K
        
            
class Vanilla_on_NonTraded:
    def __init__(self, underlying, K, T, call, MC_setup_max = 10000):
        self.underlying = underlying
        self.K = K
        self.call = call
        if self.call:
            def payoff_func(X):
                return np.maximum(X.iloc[:,-1] - K, 0)
        else:
            def payoff_func(X):
                return np.maximum(K - X.iloc[:,-1], 0)
        self.payoff_func = payoff_func
        self.T = T
        self.MC_setup_max = MC_setup_max
        self.MC_setup = self.underlying.simulate_together_Q(MC_setup_max, self.T)
        self.m = 1e-3
        
    def payoff_special(self, X_t, X0_nt):   
        X0_t = X_t.iloc[0,0]
        mu_t, sigma_t = self.underlying.underlying_t.mu, self.underlying.underlying_t.sigma
        mu_nt, sigma_nt = self.underlying.mu, self.underlying.sigma
        B_T = (np.log(X_t.iloc[:,-1] / X0_t) - (mu_t - 0.5 * sigma_t**2)*self.T) / sigma_t
        r = self.underlying.r
        rho = self.underlying.rho
        dP_dQ = np.exp((mu_t - r) * B_T / sigma_t + (0.5 * self.T * ((mu_t - r) / sigma_t) ** 2))
        a = 1
        b = 2 * sigma_nt ** 2 * self.T * (1 - rho ** 2) - 2 * np.log(X0_nt) - 2 * mu_nt * self.T + sigma_nt ** 2 * self.T - 2 * rho * sigma_nt * B_T
        c = np.log(self.m / dP_dQ  * sigma_nt * np.sqrt(2 * np.pi * self.T * (1 - rho ** 2))) * (2 * sigma_nt ** 2 * self.T * (1 - rho ** 2 )) + (-np.log(X0_nt) - mu_nt * self.T + sigma_nt ** 2 * self.T * 0.5 - rho * sigma_nt * B_T) ** 2 
        delta = b**2 - 4 * a * c
        sign = (2 * self.call * 1 - 1)
        x = sign * np.exp(( -b + sign * np.sqrt(abs(delta))) / (2 * a )) - sign * self.K
        x = x * (x >= 0) * (1 if self.call else (x < self.K))
        if self.call:
            cdf_x = norm.cdf((np.log((self.K + x) / X0_nt) - mu_nt * self.T + 0.5 * sigma_nt ** 2 * self.T - sigma_nt * rho * B_T)/ (sigma_nt * np.sqrt(self.T * (1 - rho ** 2))))
            cdf_delta = norm.cdf((np.log((self.K + 0.0001) / X0_nt) - mu_nt * self.T + 0.5 * sigma_nt ** 2 * self.T - sigma_nt * rho * B_T)/ (sigma_nt * np.sqrt(self.T * (1 - rho ** 2))))
        else:
            cdf_x = 1 - norm.cdf((np.log((self.K - x) / X0_nt) - mu_nt * self.T + 0.5 * sigma_nt ** 2 * self.T - sigma_nt * rho * B_T)/ (sigma_nt * np.sqrt(self.T * (1 - rho ** 2))))
            cdf_delta = 1 - norm.cdf((np.log((self.K - 0.0001) / X0_nt) - mu_nt * self.T + 0.5 * sigma_nt ** 2 * self.T - sigma_nt * rho * B_T)/ (sigma_nt * np.sqrt(self.T * (1 - rho ** 2))))            
        diff_x = dP_dQ * cdf_x - self.m * x
        diff_delta = dP_dQ * cdf_delta - self.m * 0.0001
        return x * (diff_delta <= diff_x) * (delta >= 0)

    def reset_MC_setup(self):
        self.MC_setup = self.underlying_t.simulate_Q(self.MC_setup_max, self.T)
    def get_MC_price(self, X0_rel_t, X0_rel_nt, t = 0, n_sims = 10000):
        if t > self.T:
            raise Exception(f'{t}> {self.T}: Pricing moment cannot exceed option expirancy moment T={self.T}')
        if self.MC_setup_max < n_sims:
            self.MC_setup_max = n_sims
            self.reset_MC_setup()
        discount = np.exp(-self.underlying.r * (self.T - t)) 
        [B_full_t, sims_full_t],  [B_full_nt, sims_full_nt] = self.MC_setup
        final_index = int(round(self.underlying.values_per_year * (self.T - t) + 1))
        B_t, sims_t = B_full_t.iloc[:n_sims,:final_index], sims_full_t.iloc[:n_sims,:final_index]
        #B_nt, sims_nt = B_full_nt.iloc[:n_sims,:final_index], sims_full_nt.iloc[:n_sims,:final_index]
        payoffs = self.payoff_special((X0_rel_t * sims_t), X0_rel_nt)
        payoffs_mean = payoffs.mean()
        MC_B = B_t.iloc[:,-1].mean()
        rho = np.sum((payoffs-payoffs_mean)*(B_t.iloc[:,-1]-0))/(payoffs.shape[0]-1)
        price = payoffs_mean - rho/self.T * (MC_B - 0)
        return discount * price
    def get_MC_delta(self, X0_rel_t, X0_rel_nt, t = 0, dX = 10, n_sims = 10000):
        price_minus = self.get_MC_price(X0_rel_t, X0_rel_nt - dX, t, n_sims)
        price_plus = self.get_MC_price(X0_rel_t, X0_rel_nt + dX, t, n_sims)
        delta = (price_plus - price_minus)/(2*dX)            
        return delta

    def set_m(self, V0, X0_t, X0_nt, precision=0.1, max_iterations=100):
        m_curr_top = 0.2
        m_curr_bot = 0.000001

        i = 1

        while self.get_MC_price(X0_t, X0_nt, m_curr_top) > V0:
            if i > max_iterations:
                raise Exception("Couldn't find m from given V0 during " + str(max_iterations) + " iterations")
            m_curr_bot = m_curr_top
            m_curr_top += 0.1
            i += 1

        m = (m_curr_top + m_curr_bot) / 2

        while True:
            if i > max_iterations:
                raise Exception("Couldn't find m from given V0 during " + str(max_iterations) + " iterations")

            V0_curr = self.get_MC_price(X0_t, X0_nt, m)

            if abs(V0 - V0_curr) < precision:
                self.m = m
                break

            elif V0_curr < V0:
                m_curr_top = m
                m = (m + m_curr_bot) / 2
                i += 1

            else:
                m_curr_bot = m
                m = (m_curr_top + m) / 2
                i += 1
