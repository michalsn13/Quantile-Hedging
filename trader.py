import numpy as np
from quantile_hedging_calculator import *

class Trader:
    def __init__(self, initial_capital):
        self.money = initial_capital
        self.delta = 0
    def update_portfolio(self, option, underlying_price, t, qh_boundary = None):
        if 'Vanilla' in str(option.__class__):
            if qh_boundary:
                delta_curr = option.get_delta(underlying_price, t, qh_boundary)
            else:
                delta_curr = option.get_delta(underlying_price, t)
        else:
            delta_curr = option.get_MC_delta(underlying_price, t)
        self.money = self.money - (delta_curr - self.delta) * underlying_price 
        self.delta = delta_curr
    def simulate_hedging(self, option, reality, update_freq = 1, limited_capital = False, verbose = False):
        if limited_capital:
            new_payoff_func,objective_func, qh_boundary = payoff_from_v0(option, self.money, float(reality[0]))
            if verbose:
                print(f'Quantile Hedging with V0={self.money:.2f} should result success probability = {objective_func[0]:.4} and success ratio = {objective_func[1]:.44}')
            old_payoff_func = option.payoff_func
            setattr(option, 'payoff_func', new_payoff_func)
        else:
            objective_func = (1,1)
            qh_boundary = None
        values_per_expirance = option.underlying.values_per_year * option.T
        money_historical = []
        delta_historical = []
        update_days = np.arange(0, values_per_expirance , update_freq)
        for num in range(values_per_expirance):
            if num in update_days:
                t = num / option.underlying.values_per_year
                self.update_portfolio(option, float(reality[num]), t, qh_boundary)
                if verbose:
                    print(f'Portfolio update at t={t}! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
            self.money *= np.exp(option.underlying.r / option.underlying.values_per_year)
            money_historical.append(self.money)
            delta_historical.append(self.delta)
        #get rid of underlying
        self.money = self.money + self.delta * float(reality[values_per_expirance])
        self.delta = 0
        delta_historical.append(self.delta)
        if verbose:
            print(f'All of held underlying {"sold" if self.delta >=0 else "repaid"}! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
        if limited_capital:
            setattr(option, 'payoff_func', old_payoff_func)
        #payoff
        payoff = float(option.payoff_func(reality))
        self.money -= payoff
        money_historical.append(self.money)
        if verbose:
            print(f'Payoff of {payoff:.2f} paid to the option owner! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
        
        return money_historical, delta_historical, objective_func