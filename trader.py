import numpy as np
from quantile_hedging_calculator import *

class Trader:
    def __init__(self, initial_capital):
        self.money = initial_capital
        self.delta = 0
    def update_portfolio(self, option, underlying_price, t, qh_boundary = None):
        if str(option.__class__).split("'")[1] == 'option.Vanilla':
            if qh_boundary:
                delta_curr = option.get_delta(underlying_price, t, qh_boundary)
            else:
                delta_curr = option.get_delta(underlying_price, t)
            self.money = self.money - (delta_curr - self.delta) * underlying_price 
        elif str(option.__class__).split("'")[1] == 'option.Vanilla_on_NonTraded':
            delta_curr = option.get_MC_delta(*underlying_price, t)
            self.money = self.money - (delta_curr - self.delta) * underlying_price[0]
        else:
            delta_curr = option.get_MC_delta(underlying_price, t)
            self.money = self.money - (delta_curr - self.delta) * underlying_price 
        self.delta = delta_curr
    def simulate_hedging(self, option, reality, update_freq = 1, mode = 'standard', recalculate_m = True, verbose = False, invest_saved_money = (False,0)):
        if mode == 'quantile_traded':
            if invest_saved_money[0]:
                saved_money = invest_saved_money[1] - self.money
            new_payoff_func,objective_func, qh_boundary = payoff_from_v0(option, self.money, float(reality[0]))
            if verbose:
                print(f'Quantile Hedging with V0={self.money:.2f} should result success probability = {objective_func[0]:.4} and success ratio = {objective_func[1]:.44}')
            old_payoff_func = option.payoff_func
            setattr(option, 'payoff_func', new_payoff_func)
        elif mode == 'quantile_nontraded':
            if invest_saved_money[0]:
                saved_money = invest_saved_money[1] - self.money
            if recalculate_m:
                old_m = option.m
                option.set_m(V0 = self.money, X0_t = reality[0].iloc[0,0], X0_nt = reality[1].iloc[0,0])
            objective_func = (1,1)
            qh_boundary = None
        else:
            if invest_saved_money[0]:
                raise Exception ('In full hedging there is no saved money to be invested')
            objective_func = (1,1)
            qh_boundary = None
        values_per_expirance = option.underlying.values_per_year * option.T
        money_historical = []
        delta_historical = []
        update_days = np.arange(0, values_per_expirance , update_freq)
        for num in range(values_per_expirance):
            reality_today = [float(reality[0][num]), float(reality[1][num])]  if mode == 'quantile_nontraded' else float(reality[num])
            if num in update_days:
                t = num / option.underlying.values_per_year
                self.update_portfolio(option, reality_today, t, qh_boundary)
                if verbose:
                    print(f'Portfolio update at t={t}! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
            self.money *= np.exp(option.underlying.r / option.underlying.values_per_year)
            money_historical.append(self.money)
            delta_historical.append(self.delta)
        #get rid of underlying
        reality_today = float(reality[0][num])  if mode == 'quantile_nontraded' else float(reality[num])
        self.money = self.money + self.delta * reality_today
        self.delta = 0
        delta_historical.append(self.delta)
        if verbose:
            print(f'All of held underlying {"sold" if self.delta >=0 else "repaid"}! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
        if mode == 'quantile_traded':
            setattr(option, 'payoff_func', old_payoff_func)
        elif mode == 'quantile_nontraded' and recalculate_m:
            setattr(option, 'm', old_m)            
        #payoff
        payoff = float(option.payoff_func(reality[1])) if mode == 'quantile_nontraded' else float(option.payoff_func(reality))
        self.money -= payoff

        if invest_saved_money[0]:
            self.money += saved_money*np.exp(option.underlying.r*option.T)

        money_historical.append(self.money)
        if verbose:
            print(f'Payoff of {payoff:.2f} paid to the option owner! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
        
        return money_historical, delta_historical, objective_func