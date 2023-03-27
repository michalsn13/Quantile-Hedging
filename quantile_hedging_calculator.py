import pandas as pd
import numpy as np

def payoff_from_v0(option, init_capital, X0, n_sims = 10000):
    
    W, X = option.underlying.simulate_P(n_sims, T=option.T)
    full = pd.concat([W.iloc[:,-1],X.iloc[:,-1]], axis = 1)
    full.columns = ['W','X']
    if 'Vanilla' in str(option.__class__):
        BS_Price = option.get_price(X0, 0)
    else:
        BS_Price = option.get_MC_price(X0, 0, n_sims)        
    if BS_Price < init_capital:
        raise Exception('With that initial capital you can perform full hedging')
    full['dP_dQstar'] = (np.exp(
        (option.underlying.mu - option.underlying.r) * W.iloc[:,-1] / option.underlying.sigma +
        (0.5 * ((option.underlying.mu - option.underlying.r) / option.underlying.sigma) ** 2 +
         option.underlying.r) * option.T) * BS_Price / option.payoff_func(X0*X))
    full = full.sort_values('dP_dQstar', ascending = False)
    full['dQstar_dP'] = 1 / full['dP_dQstar']
    
    hedge_prob = init_capital / BS_Price
    index = (full['dQstar_dP'].cumsum() / n_sims <= hedge_prob).sum()
    success_prob = index/n_sims
    if index == -1:
        raise Exception('You cannot perform any hedging with so little initial capital')

    c = (X0*full['X']).iloc[index - 1]
    
    old_payoff = option.payoff_func
    if (c < (X0*full['X']).iloc[-1]) or (c > (X0*full['X']).iloc[0]):
        def payoff_1A(X):
            return old_payoff(X)*(X.iloc[:,-1] <= c)
    else:
        def payoff_1A(X):
            return old_payoff(X)*(X.iloc[:,-1] >= c)
    return (payoff_1A, success_prob, c)

def payoff_from_prob(option, success_prob, X0, n_sims = 10000):
    if 1 <= success_prob:
        raise Exception('To hedge with 100% certainty use full hedge')

    W, X = option.underlying.simulate_P(n_sims, T=option.T)
    full = pd.concat([W.iloc[:,-1],X.iloc[:,-1]], axis = 1)
    full.columns = ['W','X']
    if 'Vanilla' in str(option.__class__):
        BS_Price = option.get_price(X0, 0)
    else:
        BS_Price = option.get_MC_price(X0, 0, n_sims)   
        
    full['dP_dQstar'] = (np.exp(
        (option.underlying.mu - option.underlying.r) * W.iloc[:,-1] / option.underlying.sigma +
        (0.5 * ((option.underlying.mu - option.underlying.r) / option.underlying.sigma) ** 2 +
         option.underlying.r) * option.T) * BS_Price / option.payoff_func(X0*X))
    full = full.sort_values('dP_dQstar', ascending = False)
    full['dQstar_dP'] = 1 / full['dP_dQstar']

    index = round(success_prob * n_sims)
    init_capital = full['dQstar_dP'].iloc[:index].sum()/n_sims * BS_Price
    c = (X0 * full['X']).iloc[index - 1]
    
    if (c < (X0*full['X']).iloc[-1]) or (c > (X0*full['X']).iloc[0]):
        def payoff_1A(X):
            return old_payoff(X)*(X.iloc[:,-1] <= c)
    else:
        def payoff_1A(X):
            return old_payoff(X)*(X.iloc[:,-1] >= c)
    return (payoff_1A, init_capital, c)