import numpy as np

class Trader:
    def __init__(self, money, delta = 0):
        self.money = money
        self.delta = delta
    def update_portfolio(self, option, underlying_price, t):
        delta_curr = option.get_MC_delta(underlying_price, t)
        self.money -= (delta_curr - self.delta) * underlying_price
        self.delta = delta_curr
    def full_hedge(self, option, reality, update_freq = 1, verbose = False):
        reality_flat = reality.flatten()
        money_historical = [self.money]
        delta_historical = [self.delta]
        update_days = np.arange(0, reality_flat.shape[0], update_freq)
        for num in update_days:
            t = num / len(update_days) * option.T
            self.update_portfolio(option, reality_flat[num], t)
            money_historical.append(self.money)
            delta_historical.append(self.delta)
            if verbose:
                print(f'Portfolio update at t={t}! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
        #get rid of underlying
        self.money += self.delta * reality_flat[-1]
        self.delta = 0
        money_historical.append(self.money)
        delta_historical.append(self.delta)
        if verbose:
            print(f'All of held underlying {"sold" if self.delta >=0 else "repaid"}! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
        #payoff
        payoff = option.payoff_func(reality)[0]
        self.money -= payoff
        money_historical.append(self.money)
        if verbose:
            print(f'Payoff of {payoff:.2f} paid to the option owner! Current status\n\tMONEY: {self.money:.2f}\n\tUNDERLYING: {self.delta:.4f}')
        return money_historical, delta_historical