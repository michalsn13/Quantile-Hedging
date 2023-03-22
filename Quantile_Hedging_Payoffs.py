def payoff_from_v0(option, init_capital, X0, n_sims = 10000):

    WX = Akcja.simulate_P(n_sims, T=option.T)
    order = WX[1][option.underlying.values_per_year * option.T].argsort()
    W_sorted = WX[0].iloc[order, :]
    X_sorted = WX[1].iloc[order, :]

    Price = option.get_MC_price(X0, 0, n_sims, 'crude')

    if Price <= init_capital:
        raise Exception('With that initial capital you can perform full hedging')

    dP_dQstar = (np.exp(
        (option.underlying.mu - option.underlying.r) * W_sorted[option.underlying.values_per_year * option.T] / option.underlying.sigma +
        (0.5 * ((option.underlying.mu - option.underlying.r) / option.underlying.sigma) ** 2 +
         option.underlying.r) * option.T) * Price / option.payoff_func(X_sorted))

    dQstar_dP = 1 / dP_dQstar

    hedge_prob = init_capital / Price
    index = sum(np.cumsum(dQstar_dP) / n < hedge_prob) - 1

    if index == -1:
        raise Exception('You cannot perform any hedging with that initial capital')

    c = X0*X_sorted.iloc[index, -1]

    old_payoff = option.payoff_func

    def payoff_1A(X):
        return old_payoff(X)*(X.iloc[:,-1] <= c)

    return (payoff_1A, hedge_prob)

def payoff_from_prob(option, hedge_prob, X0, n_sims = 10000):
    if 1 <= hedge_prob:
        raise Exception('To hedge with 100% certainty use full hedge')

    WX = Akcja.simulate_P(n_sims, T=option.T)
    order = WX[1][option.underlying.values_per_year * option.T].argsort()
    W_sorted = WX[0].iloc[order, :]
    X_sorted = WX[1].iloc[order, :]

    Price = option.get_MC_price(X0, 0, n_sims, 'crude')

    init_capital = Price * hedge_prob

    dP_dQstar = (np.exp(
        (option.underlying.mu - option.underlying.r) * W_sorted[option.underlying.values_per_year * option.T] / option.underlying.sigma +
        (0.5 * ((option.underlying.mu - option.underlying.r) / option.underlying.sigma) ** 2 +
         option.underlying.r) * option.T) * Price / option.payoff_func(X_sorted))

    dQstar_dP = 1 / dP_dQstar

    index = sum(np.cumsum(dQstar_dP) / n < hedge_prob) - 1

    if index == -1:
        raise Exception('Chosen probability is too low to perform any hedging')

    c = X0*X_sorted.iloc[index, -1]

    def payoff_1A(X):
        return option.payoff_func(X) * (X.iloc[:, -1] <= c)

    return (payoff_1A, init_capital)

#Tworzenie parametrów i akcji/opcji
n = 10000
mu = 0.06
sigma =  0.1
r = 0.05
T = 1
K = 80
X0 = 80

Akcja = Underlying(mu, sigma, r)
Opcja = Option(Akcja, lambda X: payoff_call(X, K), T)

#Wywołanie funkcji tworzących payoff i obliczających p-stwo hedge'u/ potrzebny kapitał początkowy
QH_po_1 = payoff_from_v0(Opcja, 4, X0)
QH_po_2 = payoff_from_prob(Opcja, 0.9, X0)

### Wizualizacja wyników dla opcji call:

#Prawdopodobieństwo hedgeu przy V0 = 4
QH_po_1[1]

#Kapitał początkowy potrzebny na 80% hedge'u
QH_po_2[1]

#Symulacja przykładowych trajektorii
WX = Akcja.simulate_P(n,T)

#Scatter plot payoffów przykładowych trajektorii, w podejściu initial capital
plt.scatter(X0*WX[1].iloc[:,-1], QH_po_1[0](X0*WX[1]))
plt.show()

#Plot funkcji payoffu w podejściu initial capital
S = pd.DataFrame(columns=["S"])
S["S"] = np.arange(1,130)
plt.plot(S["S"],QH_po_1[0](S))
plt.show()

#Scatter plot payoffów przykładowych trajektorii, w podejściu hedge probability
plt.scatter(X0*WX[1].iloc[:,-1], QH_po_2[0](X0*WX[1]))
plt.show()

#Plot funkcji payoffu w podejściu hedge probability
S = pd.DataFrame(columns=["S"])
S["S"] = np.arange(1,130)
plt.plot(S["S"],QH_po_2[0](S))
plt.show()


### Wizualizacja wyników dla opcji Put:
Akcja = Underlying(mu, sigma, r)
Opcja = Option(Akcja, lambda X: payoff_put(X, K), T)
QH_po_1 = payoff_from_v0(Opcja, 0.5, X0)
QH_po_2 = payoff_from_prob(Opcja, 0.5, X0)

#Prawdopodobieństwo hedgeu przy V0 = 0.5
QH_po_1[1]
#Kapitał początkowy potrzebny na 50% hedge'u
QH_po_2[1]

#Symulacja przykładowych trajektorii
WX = Akcja.simulate_P(n,T)

#Scatter plot payoffów przykładowych trajektorii, w podejściu initial capital
plt.scatter(X0*WX[1].iloc[:,-1], QH_po_1[0](X0*WX[1]))
plt.show()

#Plot funkcji payoffu w podejściu initial capital
S = pd.DataFrame(columns=["S"])
S["S"] = np.arange(1,130)
plt.plot(S["S"],QH_po_1[0](S))
plt.show()

#Scatter plot payoffów przykładowych trajektorii, w podejściu hedge probability
plt.scatter(X0*WX[1].iloc[:,-1], QH_po_2[0](X0*WX[1]))
plt.show()

#Plot funkcji payoffu w podejściu hedge probability
S = pd.DataFrame(columns=["S"])
S["S"] = np.arange(1,130)
plt.plot(S["S"],QH_po_2[0](S))
plt.show()


### Delta Hedging w Quiantile Hedgingu:
repeat = 100 # = 1000
underlying = Underlying(mu, sigma, r, 250)
_, reality = underlying.simulate_P(repeat, T)
vanilla_call = Option(underlying, lambda X: payoff_call(X, K), T)
new_payoff = payoff_from_v0(vanilla_call,4,X0)[0]
vanilla_call.payoff_func = new_payoff

money_time_call = pd.DataFrame(np.zeros(reality.shape))
delta_time_call = pd.DataFrame(np.zeros(reality.shape))

for i in tqdm(range(repeat)):
    trader = Trader(money = 4)
    money, delta = trader.full_hedge(vanilla_call, X0*reality.iloc[[i],:], update_freq = 1)
    money_time_call.loc[i] = money
    delta_time_call.loc[i] = delta

#Wykres Trajektorii cen akcji
(X0*reality).T.plot(legend = False)
plt.show()

#Wykres posiadanej gotówki w portfelu
(money_time_call).T.plot(legend = False)
plt.show()

#Wykres posiadanej liczby aktywa
(delta_time_call).T.plot(legend = False)
plt.show()

#Histogram wyników
plt.hist(money_time_call.iloc[:,-1], bins = 20)
plt.show()