import numpy as np
import scipy.linalg as linalg

b=np.array([[24,50,34,45,43,29,31,30,36,33,39,31,24,36,29,43,50,40,42,35,31,48,42,38,40,30,25,33,30,25]]).T
a=np.array([[1]*len(b),              # add a series of ones for incept beta0
            [15,16,13,20,20,18,20,10,13,16,19,20,10,12,10,15,12,14,11,13,15,19,13,13,18,10,19,11,13,14],
            [12,28,19,26,22,14,10,11,16,15,16,16,12,22,13,23,28,20,20,15,10,30,16,24,20,16,10,18,12,15]]).T
			
#Using QR decomposition to solve linear regression
Q, R = linalg.qr(a) # QR decomposition with qr function
_,row=a.shape      # find the number of rows
R=R[:row]
y = np.dot(Q.T, b)[:row] # Let y=Q'.B using matrix multiplication
x = linalg.solve(R, y) # Solve Rx=y
print('Beta0 is {:.4f} \nBeta1 is {:.4f} \nBeta2 is {:.4f}'.format(float(x[0]),float(x[1]),float(x[2])))

# Predict final result
x1=12
x2=16
final=x[0]+x[1]*x1+x[2]*x2
print('Given assignment mark {} and Midterm mark {}, \npredicted final mark is {:.4f}'.format(x1,x2,float(final)))

# Predict final result
x1=12
x2=16
final=x[0]+x[1]*x1+x[2]*x2
print('Given assignment mark {} and Midterm mark {}, \npredicted final mark is {:.4f}'.format(x1,x2,float(final)))

#Computed total marks
Total_Mark=(x1/20*0.2+x2/30*0.3+final/50*0.5)*100
print('Predicted total mark is:',float(Total_Mark))

#Using statsmodel to verify the result
import statsmodels.api as sm
x=np.array([[15,16,13,20,20,18,20,10,13,16,19,20,10,12,10,15,12,14,11,13,15,19,13,13,18,10,19,11,13,14],
           [12,28,19,26,22,14,10,11,16,15,16,16,12,22,13,23,28,20,20,15,10,30,16,24,20,16,10,18,12,15]]).T
y=np.array([[24,50,34,45,43,29,31,30,36,33,39,31,24,36,29,43,50,40,42,35,31,48,42,38,40,30,25,33,30,25]]).T
X=sm.add_constant(x)
model = sm.OLS(y,X)
results = model.fit()
results.params