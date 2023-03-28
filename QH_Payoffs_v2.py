import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class Underlying:
    def __init__(self, mu, sigma, r, values_per_year = 365):
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.values_per_year = values_per_year
    def simulate_P(self, size, T):
        price_moments = np.arange(0,self.values_per_year * T + 1)
        Sigma=1/self.values_per_year*np.minimum(np.tile(price_moments,(len(price_moments),1)),np.tile(price_moments.reshape(-1,1),(1,len(price_moments))))
        B = pd.DataFrame(np.random.multivariate_normal(size=size, mean= np.zeros(len(price_moments)), cov = Sigma))
        sims = np.exp((self.mu - 0.5 * self.sigma**2) * price_moments / self.values_per_year + self.sigma * B)
        return (B,sims)
    def simulate_Q(self, size, T):
        price_moments = np.arange(0,self.values_per_year * T + 1)
        Sigma=1/self.values_per_year*np.minimum(np.tile(price_moments,(len(price_moments),1)),np.tile(price_moments.reshape(-1,1),(1,len(price_moments))))
        B = pd.DataFrame(np.random.multivariate_normal(size=size, mean= np.zeros(len(price_moments)), cov = Sigma))
        sims = np.exp((self.r - 0.5 * self.sigma**2) * price_moments / self.values_per_year + self.sigma * B)
        return (B,sims)

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

def payoff_call(X, K):
    return np.maximum(X.iloc[:,-1]- K, 0)
def payoff_put(X, K):
    return np.maximum(K - X.iloc[:,-1], 0)

#############################

def payoff_from_v0(option, init_capital, X0, n_sims=10000):

    if 'Vanilla' in str(option.__class__):
        BS_Price = option.get_price(X0, 0)
    else:
        BS_Price = option.get_MC_price(X0, 0, n_sims)
    if BS_Price < init_capital:
        raise Exception('With that initial capital you can perform full hedging')

    W, X = option.underlying.simulate_P(n_sims, T=option.T)

    dP_dQstar = (np.exp(
        (option.underlying.mu - option.underlying.r) * W[
            option.underlying.values_per_year * option.T] / option.underlying.sigma +
        (0.5 * ((option.underlying.mu - option.underlying.r) / option.underlying.sigma) ** 2 +
         option.underlying.r) * option.T) * BS_Price / option.payoff_func(X0 * X))

    order = (-dP_dQstar).argsort()
    dP_dQstar_sorted = dP_dQstar.iloc[order]
    W_sorted = W.iloc[order, :] #Chyba do wyrzucenia
    X_sorted = X.iloc[order, :]
    dQstar_dP_sorted = 1 / dP_dQstar_sorted

    alpha = init_capital / BS_Price

    if option.underlying.mu <= option.underlying.sigma**2:

        index = sum(np.cumsum(dQstar_dP_sorted) / n_sims < alpha)
        success_prob = index / n_sims
        if index == 0:
            raise Exception('You cannot perform any hedging with so little initial capital')

        c = X0 * X_sorted.iloc[index - 1, -1]
        old_payoff = option.payoff_func

        def payoff_1A(X):
            return old_payoff(X) * (X.iloc[:, -1] <= c)

        return (payoff_1A, success_prob, (c,))

    else:
        index1 = sum(np.cumsum(dQstar_dP_sorted) / n_sims < 0.5*alpha)
        index2 = n_sims - sum(np.cumsum(np.flip(dQstar_dP_sorted)) / n_sims < 0.5*alpha)
        success_prob = 1 - (index2-index1)/ n_sims

        c1 = X0 * X_sorted.iloc[index1 - 1, -1]
        c2 = X0 * X_sorted.iloc[index2, -1]
        old_payoff = option.payoff_func

        def payoff_1A(X):
            # Dodawanie to logiczny operator OR w data frame'ach z pandasa z wartościami booleanskimi
            return old_payoff(X) * ((X.iloc[:, -1] <= c1) + (X.iloc[:, -1] >= c2))
        return (payoff_1A, success_prob, (c1,c2))


def payoff_from_prob(option, success_prob, X0, n_sims=10000):
    if 1 <= success_prob:
        raise Exception('To hedge with 100% certainty use full hedge')

    W, X = option.underlying.simulate_P(n_sims, T=option.T)

    if 'Vanilla' in str(option.__class__):
        BS_Price = option.get_price(X0, 0)
    else:
        BS_Price = option.get_MC_price(X0, 0, n_sims)

    dP_dQstar = (np.exp(
        (option.underlying.mu - option.underlying.r) * W[
            option.underlying.values_per_year * option.T] / option.underlying.sigma +
        (0.5 * ((option.underlying.mu - option.underlying.r) / option.underlying.sigma) ** 2 +
         option.underlying.r) * option.T) * BS_Price / option.payoff_func(X0 * X))

    order = (-dP_dQstar).argsort()
    dP_dQstar_sorted = dP_dQstar.iloc[order]
    W_sorted = W.iloc[order, :]  # Chyba do wyrzucenia
    X_sorted = X.iloc[order, :]
    dQstar_dP_sorted = 1 / dP_dQstar_sorted


    if option.underlying.mu <= option.underlying.sigma**2:
        index = int(success_prob * n_sims)
        init_capital = dQstar_dP_sorted[:index].sum() / n_sims * BS_Price

        c = X0 * X_sorted.iloc[index - 1, -1]

        def payoff_1A(X):
            return option.payoff_func(X) * (X.iloc[:, -1] <= c)

        return (payoff_1A, init_capital, (c,))

    else:
        d = int(n_sims-success_prob*n_sims)
        v = np.flip(np.cumsum(dQstar_dP_sorted)[:(n_sims-d)])
        w = np.cumsum(np.flip(dQstar_dP_sorted))[:(n_sims-d)]

        offset = np.argmin(abs(np.array(v)/np.array(w)-1))
        index1 = n_sims - d - offset
        index2 = n_sims - offset
        init_capital = BS_Price*(dQstar_dP_sorted[:index1].sum() + dQstar_dP_sorted[:index1].sum())/n_sims

        c1 = X0 * X_sorted.iloc[index1 - 1, -1]
        c2 = X0 * X_sorted.iloc[index2, -1]
        old_payoff = option.payoff_func

        def payoff_1A(X):
            # Dodawanie to logiczny operator OR w data frame'ach z pandasa z wartościami booleanskimi
            return old_payoff(X) * ((X.iloc[:, -1] <= c1) + (X.iloc[:, -1] >= c2))

        return (payoff_1A, init_capital, (c1,c2))


#Showcase przypadku z c1 i c2
mu = 0.01
sigma =  0.05
r = 0.05
T = 1
X0 = 100
repeat = 100
K = 80
n_sims = 1000

underlying = Underlying(mu, sigma, r, 250)
option = Option(underlying, lambda X: payoff_call(X, K), T)

#Cena
option.get_MC_price(X0)
#Przypadek V0 -> Payoff, Wiec bierzemy V0=17
Res1 = payoff_from_v0(option,17,X0)

#Pstwo sukcesu:
Res1[1]
Res1[2][0] #c1
Res1[2][1] #c2

Y1 = pd.DataFrame(np.arange(70,141,0.1))
plt.plot(np.arange(70,141,0.1),np.array(Res1[0](Y1)))
plt.show()

#Przypadek pstwo -> payoff, bierzemy 80%
Res2 = payoff_from_prob(option,0.8,X0)

#Pstwo kapitał początkowy:
Res2[1]
Res2[2][0] #c1
Res2[2][1] #c2

Y2 = pd.DataFrame(np.arange(70,141,0.1))
plt.plot(np.arange(70,141,0.1),np.array(Res2[0](Y2)))
plt.show()


#### Puta MAdre
mu = 0.01
sigma =  0.05
r = 0.05
T = 1
X0 = 100
repeat = 100
K = 120
n_sims = 1000

underlying = Underlying(mu, sigma, r, 250)
option = Option(underlying, lambda X: payoff_put(X, K), T)

#Cena
option.get_MC_price(X0)
#Przypadek V0 -> Payoff, Wiec bierzemy V0=11
Res3 = payoff_from_v0(option,11,X0)

#Pstwo sukcesu:
Res3[1]
Res3[2][0] #c1
Res3[2][1] #c2

Y3 = pd.DataFrame(np.arange(70,141,0.1))
plt.plot(np.arange(70,141,0.1),np.array(Res3[0](Y3)))
plt.show()

# i z pstwa
Res4 = payoff_from_prob(option,0.8,X0)

#Pstwo sukcesu:
Res4[1]
Res4[2] #c1
Res4[3] #c2

Y4 = pd.DataFrame(np.arange(70,141,0.1))
plt.plot(np.arange(70,141,0.1),np.array(Res4[0](Y4)))
plt.show()
