
from european import European

class Strategy:
    def __init__(self, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, call):
        self.S = spot_price
        self.K = strike_price
        self.T = time_to_maturity
        self.r = risk_free_rate
        self.sigma = volatility
        self.call = call

