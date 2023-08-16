import numpy as np
from scipy.stats import norm

class European:
    def __init__(self, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, call):
        self.S = spot_price
        self.K = strike_price
        self.T = time_to_maturity
        self.r = risk_free_rate
        self.sigma = volatility
        self.call = call

    def black_scholes(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.call:
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.S * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def binomial(self, n=252):
        dt = self.T / n
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        q = 1 - p

        stock_price = [[0 for j in range(i + 1)] for i in range(n + 1)]
        stock_price[0][0] = self.S

        for i in range(1, n + 1):
            for j in range(i + 1):
                if j == 0:
                    stock_price[i][j] = stock_price[i - 1][j] * u
                else:
                    stock_price[i][j] = stock_price[i - 1][j - 1] * d

        option_value = [[0 for j in range(i + 1)] for i in range(n + 1)]

        for j in range(n + 1):
            if self.call:
                option_value[n][j] = max(stock_price[n][j] - self.K, 0)
            else:
                option_value[n][j] = max(self.K - stock_price[n][j], 0)

        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                option_value[i][j] = np.exp(-self.r * dt) * (p * option_value[i + 1][j] + q * option_value[i + 1][j + 1])

        return option_value[0][0]

    def trinomial(self, n=252):
        dt = self.T / n
        up_factor = np.exp(self.sigma * np.sqrt(2 * dt))
        down_factor = 1 / up_factor

        p_u = ((np.exp(self.r * dt / 2) - np.exp(-self.sigma * np.sqrt(dt / 2))) / (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2))))**2
        p_d = ((np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(self.r * dt / 2)) / (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2))))**2
        p_m = 1 - p_u - p_d

        stock_price = [[0 for j in range(2 * i + 1)] for i in range(n + 1)]
        stock_price[0][0] = self.S

        for i in range(1, n + 1):
            for j in range(2 * i + 1):
                if j == 0:
                    stock_price[i][j] = stock_price[i - 1][j] * up_factor
                elif j == (2 * i):
                    stock_price[i][j] = stock_price[i - 1][j - 2] * down_factor
                else:
                    stock_price[i][j] = stock_price[i - 1][j - 1]

        option_value = [[0 for j in range(2 * i + 1)] for i in range(n + 1)]

        for j in range(2 * n + 1):
            option_value[n][j] = max(stock_price[n][j] - self.K, 0)

        for i in range(n - 1, -1, -1):
            for j in range(2 * i + 1):
                option_value[i][j] = np.exp(-self.r * dt) * (p_u * option_value[i+1][j] + p_m * option_value[i+1][j+1] + p_d * option_value[i+1][j+2])

        return option_value[0][0]

    def monte_carlo(self, sims=100000):
        option_value = np.zeros(sims)
        for i in range(sims):
            S_T = self.S * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T + self.sigma * np.sqrt(self.T) * np.random.normal())
            if self.call:
                payoff = max(S_T - self.K, 0)
            else:
                payoff = max(self.K - S_T, 0)
            option_value[i] = payoff * np.exp(-self.r * self.T)

        return np.mean(option_value)

    def delta(self):

        pass

    def gamma(self):
        pass

    def theta(self):
        pass

    def rho(self):
        pass

    def vega(self):
        pass


# Example usage
spot = 100
strike = 100
time = 1
rate = 0.05
vol = 0.2

option = European(spot, strike, time, rate, vol, False)

bs = option.black_scholes()
mc = option.monte_carlo()
bi = option.binomial()
tri = option.trinomial()

print("Black-Scholes Call Price:", bs, mc, bi, tri)

