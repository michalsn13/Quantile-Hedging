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
    old_payoff = option.payoff_func
    condition = option.payoff_func(X0 * X.loc[X.iloc[:,[-1]].idxmin()]).values[0] < option.payoff_func(X0 * X.loc[X.iloc[:,[-1]].idxmax()]).values[0]
    if ((option.underlying.mu - option.underlying.r) <= option.underlying.sigma**2 and condition) or ((option.underlying.mu - option.underlying.r) > option.underlying.sigma**2 and not condition):
        index = (full['dQstar_dP'].cumsum() / n_sims <= hedge_prob).sum()
        success_prob = index/n_sims
        if index == 0:
            raise Exception('You cannot perform any hedging with so little initial capital')

        c = (X0*full['X']).iloc[index - 1]
        if condition:
            X_fail = X.loc[(X0*X).iloc[:,-1] > c]
            H = old_payoff((X0*X_fail))
            VT_H = ((H - (BS_Price - init_capital) * np.exp(option.underlying.r * option.T))/H).sum() / X.shape[0]
            c1, c2 = c, 0
            def payoff_1A(X):
                return old_payoff(X)*(X.iloc[:,-1] <= c)
        else:
            X_fail = X.loc[(X0*X).iloc[:,-1] < c]
            H = old_payoff((X0*X_fail))
            VT_H = ((H - (BS_Price - init_capital) * np.exp(option.underlying.r * option.T))/H).sum() / X.shape[0]
            c1, c2 = 0, c
            def payoff_1A(X):
                return old_payoff(X)*(X.iloc[:,-1] >= c)
    else:
        index1 = (full['dQstar_dP'].cumsum() / n_sims <= hedge_prob/2).sum()
        index2 = (full['dQstar_dP'][::-1].cumsum() / n_sims <= hedge_prob/2).sum()
        success_prob = (index1 + index2)/n_sims
        if np.max([index1,index2]) == 0:
            raise Exception('You cannot perform any hedging with so little initial capital')

        c1 = (X0*full['X']).iloc[(index1 - 1)]
        c2 = (X0*full['X']).iloc[-(index2 - 1)]
        c1, c2 = np.min([c1,c2]), np.max([c1,c2])
        X_fail = X.loc[((X0*X).iloc[:,-1] > c1) & ((X0*X).iloc[:,-1] < c2)]
        H = old_payoff((X0*X_fail))
        VT_H = ((H - (BS_Price - init_capital) * np.exp(option.underlying.r * option.T))/H).sum() / X.shape[0]
        def payoff_1A(X):
            return old_payoff(X)*np.maximum((X.iloc[:,-1] <= c1), (X.iloc[:,-1] >= c2))
    success_ratio = success_prob + VT_H
    return (payoff_1A, (success_prob, success_ratio), (c1,c2))
        

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
    old_payoff = option.payoff_func
    if (option.underlying.mu - option.underlying.r) <= option.underlying.sigma**2:
        index = round(success_prob * n_sims)
        init_capital = full['dQstar_dP'].iloc[:index].sum()/n_sims * BS_Price
        c = (X0 * full['X']).iloc[index - 1]
        if (c < (X0*full['X']).iloc[-1]) or (c > (X0*full['X']).iloc[0]):
            def payoff_1A(X):
                return old_payoff(X)*(X.iloc[:,-1] <= c)
        else:
            def payoff_1A(X):
                return old_payoff(X)*(X.iloc[:,-1] >= c)
        return (payoff_1A, init_capital, (0,c))
    else:
        d = round(success_prob*n_sims)
        v = full['dQstar_dP'].cumsum()[:(d-1)].reset_index(drop = True)
        w = full['dQstar_dP'][::-1].cumsum()[:(d-1)][::-1].reset_index(drop = True)
        ratios = abs((v/w).fillna(np.infty) - 1)
        index1 = np.argmin(ratios)
        index2 = (d-1)-index1
        init_capital = BS_Price*(full['dQstar_dP'][:(index1 + 1)].sum() + full['dQstar_dP'][-index2:].sum())/n_sims

        c1 = (X0*full['X']).iloc[index1]
        c2 = (X0*full['X']).iloc[-index2]
        c1, c2 = np.min([c1,c2]), np.max([c1,c2])
        def payoff_1A(X):
            return old_payoff(X)*np.maximum((X.iloc[:,-1] <= c1), (X.iloc[:,-1] >= c2))
        return (payoff_1A, init_capital, (c1,c2))
        
