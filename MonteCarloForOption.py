#Simulate stock price paths and European option at maturity
#Calculate difference(error) between simulated option price and Black-Scholes option price
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt

class MonteCarlo_Option:
    #initialize all input variables
    def __init__(self, S,K,Sigma,R, Trading_Days,NumberOfSimulation,FrequencyPerDay,T) :
        self._S=S
        self._K=K
        self._Sigma=Sigma
        self._R=R
        self._Trading_Days=Trading_Days
        self._NoOfSimulation=NumberOfSimulation
        self._FrequencyPerDay=FrequencyPerDay
        self._NoOftrades=self._Trading_Days*self._FrequencyPerDay
        self._dt=T/(365*self._FrequencyPerDay)
        self._T=T
        
    #Method to create stock simulation paths
    def Stock_Simulation(self):
        np.random.seed(2)
        ls=[]
        for s in range(self._NoOfSimulation):
            stock=[0]*(self._NoOftrades+1)
            stock[0]=self._S
            for i in range (1, self._NoOftrades+1):
                shares=stock[i-1]*np.exp((self._R-0.5*self._Sigma**2)*self._dt
                                   +self._Sigma*sqrt(self._dt)*np.random.normal(0,1)) #standard random normal number
                stock[i]=shares
            ls.append(stock)
        SimulateList=np.array(ls)
        #print(SimulateList)
        return SimulateList
    
    #Method to plot stock paths
    def Plot_Stock(self):
        plt.figure(figsize=(10,6))
        plt.xlabel('Number Of Trades')
        plt.ylabel('Stock price')
        for i in self.Stock_Simulation():
            plt.plot(i)
        plt.title('Stock price:{} paths for {} trades'.format(self._NoOfSimulation,
                  self._NoOftrades))
        plt.show()
    
    #Option price using Black-Scholes
    def OptionCall(self): 
        d1=(np.log(self._S/self._K)
            +((self._R+(0.5*self._Sigma**2))/2)*self._T)/(self._Sigma*np.sqrt(self._T))
        d2=d1-self._Sigma*np.sqrt(self._T)
        call_price=self._S*norm.cdf(d1) - self._K*np.exp(-self._R*self._T)*norm.cdf(d2)
        print("Black Scholes option price is: {}".format(call_price))
        return call_price
    
    #Calculate simulated option price 
    def Simulated_Option(self):
        Option_payoff=np.array(self.Stock_Simulation()[:,-1]-self._K)
        option=[]
        for i in Option_payoff:
            option.append(max(i,0))
        option=np.array(option).mean()
        option=option*np.exp(-self._R*self._T)
        Option_payoff=Option_payoff*np.exp(-self._R*self._T)
        print("Simulated option payoff at maturity: {}".format(option))
        return option,Option_payoff
    
def Error():
    path=np.arange(500,10001,500)
    price=[]
    std_error=[]
    for i in path:
        call=MonteCarlo_Option(30,32,0.2,0.03,365,i,1,1)
        call_price,option=call.Simulated_Option()
        SE=np.std(option)/np.sqrt(i)
        price.append(call_price)
        std_error.append(SE)
    e={'Number of paths':path,'call price':price,'standard error':std_error}
    df_error=pd.DataFrame(e)
    print(df_error)
    return df_error
    
        
        
    
def main():
    call=MonteCarlo_Option(30,32,0.2,0.03,365,500,1,1)
    call.Stock_Simulation()
    call.Plot_Stock()
    call.OptionCall()
    call.Simulated_Option()
    error=Error()
    plt.figure(figsize=(12,10))
    plt.plot(error['Number of paths'],error['standard error'])
    plt.xlabel('Number of paths')
    plt.ylabel('Standard error')
    plt.show()
    
main()
    
    
    
    
    
    
