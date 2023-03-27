import pandas as pd
import numpy as np

def payoff_from_v0(option, init_capital, X0, n_sims = 10000):
    
    WX = option.underlying.simulate_P(n_sims, T=option.T)
    order = WX[1][option.underlying.values_per_year * option.T].argsort()
    W_sorted = WX[0].iloc[order, :]
    X_sorted = WX[1].iloc[order, :]
    if 'Vanilla' in str(option.__class__):
        BS_Price = option.get_price(X0, 0)
    else:
        BS_Price = option.get_MC_price(X0, 0, n_sims)        
    if BS_Price < init_capital:
        raise Exception('With that initial capital you can perform full hedging')

    dP_dQstar = (np.exp(
        (option.underlying.mu - option.underlying.r) * W_sorted[option.underlying.values_per_year * option.T] / option.underlying.sigma +
        (0.5 * ((option.underlying.mu - option.underlying.r) / option.underlying.sigma) ** 2 +
         option.underlying.r) * option.T) * BS_Price / option.payoff_func(X0*X_sorted))

    dQstar_dP = 1 / dP_dQstar

    hedge_prob = init_capital / BS_Price
    index = sum(np.cumsum(dQstar_dP) / n_sims < hedge_prob)
    success_prob = index/n_sims
    if index == -1:
        raise Exception('You cannot perform any hedging with so little initial capital')

    c = X0*X_sorted.iloc[index - 1, -1]
    old_payoff = option.payoff_func

    def payoff_1A(X):
        return old_payoff(X)*(X.iloc[:,-1] <= c)
    return (payoff_1A, success_prob, c)

def payoff_from_prob(option, success_prob, X0, n_sims = 10000):
    if 1 <= success_prob:
        raise Exception('To hedge with 100% certainty use full hedge')

    WX = option.underlying.simulate_P(n_sims, T=option.T)
    order = WX[1][option.underlying.values_per_year * option.T].argsort()
    W_sorted = WX[0].iloc[order, :]
    X_sorted = WX[1].iloc[order, :]

    if 'Vanilla' in str(option.__class__):
        BS_Price = option.get_price(X0, 0)
    else:
        BS_Price = option.get_MC_price(X0, 0, n_sims)   
        
    dP_dQstar = (np.exp(
        (option.underlying.mu - option.underlying.r) * W_sorted[option.underlying.values_per_year * option.T] / option.underlying.sigma +
        (0.5 * ((option.underlying.mu - option.underlying.r) / option.underlying.sigma) ** 2 +
         option.underlying.r) * option.T) * BS_Price / option.payoff_func(X0*X_sorted))

    dQstar_dP = 1 / dP_dQstar

    index = int(success_prob * n_sims)
    init_capital = dQstar_dP[:index].sum()/n_sims * BS_Price

    c = X0*X_sorted.iloc[index - 1, -1]

    def payoff_1A(X):
        return option.payoff_func(X) * (X.iloc[:, -1] <= c)

    return (payoff_1A, init_capital, c)