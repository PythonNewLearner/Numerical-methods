import pandas as pd
import datetime as dt
import numpy as np
from scipy.stats import norm

Call={'Strike':[20.00,23.00,25.00,28.00,30.00,33.00,35.00,38.00,40.00,42.00,45.00,47.00,50.00,52.00,55.00,57.50,60.00,
     62.5,65.00,67.50,70.00,75.00,80.00,85.00,90.00],
     'Call price':[37.8,34.65,32.55,29.60,26.95,24.26,23.7,20.5,16.9,16.9,11.8,9.81,7.2,5.3,3.55,2.16,1.2,0.65,0.29,0.15,0.08,
            0.04,0.03,0.02,0.02]}
Call=pd.DataFrame(Call)
DaysToExpiry=(dt.datetime(2015,1,17)-dt.datetime(2014,8,15)).days
t=YearToExpiry=DaysToExpiry/365
S=56.75
r=0.0025
Call['Stock price'],Call['Time to Maturity'],Call['Rate']=S,t,r

#Inputs: s stock price, k strike price, c call price, t time to expiry, vol volatility, r interest rate
def IV(s,k,c,t,vol,r):
    d1=(np.log(s/k)+(r+0.5*vol**2)*t)/(vol*np.sqrt(t))
    d2=(np.log(s/k)+(r-0.5*vol**2)*t)/(vol*np.sqrt(t))
    return (s*norm.cdf(d1)-k*np.exp(-r*t)*norm.cdf(d2))-c


# 1st derivative to volatility function  (Vega)
def IVPrime(s,k,c,t,vol,r):    # 1st derivative to volatility function
    return s*np.sqrt(t/(2*np.pi))*np.exp((-(np.log(s/k)+(r+vol**2/2)*t)**2)/(2*vol**2*t))

#Implement Newton's method
def NewtonRoot(s,k,c,t,r,n,x):   #x is initial guess
    for i in range(n):
        xnew=x-IV(s,k,c,t,x,r)/IVPrime(s,k,c,t,x,r)
        if abs(xnew-x)<1e-6:
            break
        x=xnew
    return xnew

#Apply Newton's method to find Implied Volatility
Call['Implied Volatility']=Call.apply(lambda x: NewtonRoot(x[2],x[0],x[1],x[3],x[4],100,0.9),axis=1)
Call

#plot
plt.figure(figsize=(12,8))
plt.plot(Call['Strike'],Call['Implied Volatility'],label='Call Implied Volatility')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.legend()
plt.show()