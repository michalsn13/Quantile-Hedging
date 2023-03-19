import numpy as np

class Option:
    def __init__(self, underlying, payoff_func, T, MC_setup_max = 10000):
        self.underlying = underlying
        self.payoff_func = payoff_func
        self.T = T
        self.MC_setup_max = MC_setup_max
        self.MC_setup = self.underlying.simulate_Q(MC_setup_max, self.T)
    def reset_MC_setup(self):
        self.MC_setup = self.underlying.simulate_Q(self.MC_setup_max, self.T)
    def get_MC_price(self, X0_rel, t = 0, n_sims = 10000, method='crude'):
        if t > self.T:
            raise Exception(f'{t}> {T}: Pricing moment cannot exceed option expirancy moment T={self.T}')
        if self.MC_setup_max < n_sims:
            self.MC_setup_max = n_sims
            self.reset_MC_setup()
        discount = np.exp(-self.underlying.r * (self.T - t)) 
        B_full, sims_full = self.MC_setup
        final_index = int((B_full.shape[1]-1) * (self.T - t) / self.T) + 1
        B, sims = B_full[:n_sims,:final_index], sims_full[:n_sims,:final_index]
        payoffs = self.payoff_func(X0_rel * sims)
        payoffs_mean = payoffs.mean()
        if method == 'crude':
            price = payoffs_mean
        elif method == 'var_control':
            MC_B = B[:,-1].mean()
            rho = np.sum((payoffs-payoffs_mean)*(B[:,-1]-MC_B))/(payoffs.shape[0]-1)
            price = payoffs_mean - rho/1 * (MC_B - 0)
        else:
            raise Exception(f'Method {method} not implemented...')
        return discount * price
    def get_MC_delta(self, X0_rel, t = 0, n_sims = 10000, dX = 10, method='crude'):
        price_minus = self.get_MC_price(X0_rel-dX, t, n_sims, method)
        price_plus = self.get_MC_price(X0_rel+dX, t, n_sims, method)
        delta = (price_plus - price_minus)/(2*dX)
        return delta