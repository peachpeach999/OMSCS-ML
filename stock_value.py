import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict


class QTrader(object):
    def __init__(self):
        self.stock_data = pd.merge(pd.read_csv('GSPC.csv', index_col='Date'), pd.read_csv('tbills.csv',index_col='Date'),\
                                   right_index=True, left_index=True).sort_index()
        self.returns = pd.DataFrame({
            'stocks': self.stock_data['Adj Close'].rolling(window=2).apply(lambda x : x[1]/x[0]-1),
            'tbills': (self.stock_data['tbill_rate']/100 + 1)**(1/52) -1 ,
        }, index = self.stock_data.index)

        ## 12 weeks of data
        ## What's the high average
        ## What's the low average
        ## If the returns is higher than high average -> State  1
        ## If the return is lower than low average -> State  -1
        ## 0

        self.returns['risk_adjusted'] = self.returns.stocks - self.returns.tbills
        self.returns['risk_adjusted_moving'] = self.returns.risk_adjusted.rolling(window=12).apply(lambda x: x.mean())
        self.returns['risk_adjusted_stdev'] = self.returns.risk_adjusted.rolling(window=12).apply(lambda x: x.std())
        self.returns['risk_adjusted_high'] = self.returns.risk_adjusted_moving + 1.5*\
            self.returns.risk_adjusted_stdev
        self.returns['risk_adjusted_low'] = self.returns.risk_adjusted_moving - 1.5*\
            self.returns.risk_adjusted_stdev
        self.returns['state'] = (self.returns.risk_adjusted > self.returns.risk_adjusted_high).astype('int')\
            - (self.returns.risk_adjusted < self.returns.risk_adjusted_low).astype('int')

    def buy_and_hold(self, dates):
        return pd.Series(1, index = dates)

    def buy_tbills(self, dates):
        return pd.Series(0, index = dates)


    def random(self,dates):
        return pd.Series(np.random.randint(-1,2, size=len(dates)), index = dates)

    def evaluate(self, holdings):
        return pd.Series(self.returns.tbills + holdings * self.returns.risk_adjusted + 1, index=holdings.index).cumprod()

    def q_holdings(self, training_index, testing_index):
        """Value iterations
        # Q-Learning
        # iterating on State, Action, Rewards, and new State/ state prime"""
        factors = pd.DataFrame({'action': 0, 'rewards': 0 , 'state':0}, index = training_index)
        q = {0: {1:0, 0:0, -1:0}}

        for i in range(100):
            last_row , last_date = None, None
            for date, row in factors.iterrows():
                return_data = self.returns.loc[date]
                if return_data.state not in q :
                    q[return_data.state]= {1:0, 0:0, -1:0}

                if last_row is None or np.isnan(return_data.state):
                    state =0
                    rewards = 0
                    action = 0
                else:
                    state = int(return_data.state)
                    if random.random() > 0.001:
                        action = max(q[state], key=q[state].get)
                    else:
                        action = random.randint(-1,1)

                    rewards = last_row.action * (return_data.stocks - return_data.tbills)

                    factors.loc[date, 'reward'] = rewards
                    factors.loc[date,'action'] = action
                    factors.loc[date, 'state'] = return_data.state

                    alpha = 1
                    discount = 0.9

                    update = alpha *(factors.loc[date,'reward'] + discount * \
                                     max(q[row.state].values()) - q[state][action])
                    if not np.isnan(update):
                        q[state][action] += update

                last_date, last_row = date , factors.loc[date]

            sharpe = self.sharpe(factors.action)
            if sharpe > 0.2 :
                break
            print("For episode {} we get an internal sharpe ratio of {}".format(i,sharpe))

        testing = pd.DataFrame({'action':0, 'state':0 },index = testing_index)
        testing['state'] = self.returns.loc[testing_index, 'state']
        testing['action'] = testing['state'].apply(lambda state: max(q[state], key=q[state].get))

        return testing.action

    def sharpe(self, holdings):
        returns = holdings * (self.returns.stocks - self.returns.tbills )
        return np.mean(returns) / np.nanstd(returns)


    def graph_portfolio(self):
        midpoint = int(len(self.returns.index)/2)
        training_index = self.returns.index[:midpoint]
        testing_index = self.returns.index[midpoint:]

        portfolios = pd.DataFrame({
            'buy_and_hold': self.buy_and_hold(testing_index),
            'buy_tbills': self.buy_tbills(testing_index),
            'random': self.random(testing_index),
            'qtrader': self.q_holdings(training_index, testing_index)
        }, index = testing_index)

        portfolio_values = pd.DataFrame({
            'buy_and_hold': self.evaluate(portfolios.buy_and_hold),
            #'buy_tbills': self.evaluate(portfolios.buy_tbills),
            #'random': self.evaluate(portfolios.random),
            'value': self.evaluate(portfolios.qtrader)
        }, index = testing_index)

        # portfolios.plot()
        portfolio_values.plot()
        plt.title("Value-Iteration: Portfolio Values")
        #plt.annotate("Buy and hold sharp ratio: {} \n QTrader: {}".format(self.sharpe(portfolios.buy_and_hold),\
                                    #self.sharpe(portfolios.qtrader)), xy = (0.25,0.95), xycoords="axes fraction")

        #plt.show()
        plt.savefig('graph.png')


if __name__ == '__main__':
    qt = QTrader()
    qt.graph_portfolio()