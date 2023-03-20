import numpy as np
from scipy.stats import norm

class Trader:
    def __init__(self, money, delta = 0):
        self.money = money
        self.delta = delta
    def update_portfolio(self, option, underlying_price, t, dt):
        delta_curr = option.get_MC_delta(underlying_price, t)
        self.money = self.money * np.exp(option.underlying.r * dt) - (delta_curr - self.delta) * underlying_price 
        self.delta = delta_curr
    def full_hedge(self, option, reality, update_freq = 1, verbose = False):
        reality_flat = reality.flatten()
        money_historical = []
        delta_historical = []
        update_days = np.arange(0, reality_flat.shape[0]-1, update_freq)
        dt = option.T / (reality_flat.shape[0]-1)
        for num in range(reality_flat.shape[0] - 1):
            if num in update_days:
                t = num * dt
                self.update_portfolio(option, reality_flat[num], t, dt)
                if verbose:
                    print(f'Portfolio update at t={t}! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
            money_historical.append(self.money)
            delta_historical.append(self.delta)
        #get rid of underlying
        self.money = self.money * np.exp(option.underlying.r * dt) + self.delta * reality_flat[-1]
        self.delta = 0
        delta_historical.append(self.delta)
        if verbose:
            print(f'All of held underlying {"sold" if self.delta >=0 else "repaid"}! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
        #payoff
        payoff = option.payoff_func(reality.T)[0]
        self.money -= payoff
        money_historical.append(self.money)
        if verbose:
            print(f'Payoff of {payoff:.2f} paid to the option owner! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
        return money_historical, delta_historical