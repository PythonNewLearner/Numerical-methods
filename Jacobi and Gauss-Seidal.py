import numpy as np

# initialize matrix (left-hand side)
array=np.array([4]*100)
a=np.diag(array)         # Create diagnal matrix

# Create input matrix as per requested
for i in range(1,len(a)):
    for j in range(i,i+1):
        a[i-1,j]=-1
        
for i in range(1,len(a)):
    for j in range(i-1,i):
        a[i,j]=-1
print(a)

#Create matrix (right-hand side)
b=np.array([2]*100)
b[0],b[-1]=3,3

#Jacobi method
(n,)=b.shape
x=np.full(n,0.5,float)   #initial value of x is 0.5
xnew=np.empty(n,float)
iterlimit=100
tolerance=1e-5

#iteration
for iteration in range(iterlimit):
    for i in range(n):
        s=0
        for j in range(n):
            if j != i:
                s += a[i,j]*x[j]
        xnew[i]=-1/a[i,i]*(s-b[i])
    if (abs(xnew-x)<tolerance).all():   #check if all element in the array is less than tolerance value
        break
    else:
        x=np.copy(xnew)
print('Jacobi method solution :\n',x, '\n','\nNumber of iterations: \n',iteration)

#Gauss-Seidal method
(n,)=b.shape
x=np.full(n,0.5,float)   #initial value of x is 0.5
xnew=np.empty(n,float)
iterlimit=100
tolerance=1e-5
#iteration
for iteration in range(iterlimit):
    for i in range(n):
        s=0
        for j in range(n):
            if j != i:
                s += a[i,j]*x[j]
        xnew=-1/a[i,i]*(s-b[i])
        xdiff=abs(xnew-x[i])
        x[i]=xnew
    if (xdiff<tolerance).all():   #check if all element in the array is less than tolerance value
        break
print('Gauss-Seidal method solution :\n',x, '\n','\nNumber of iterations: \n',iteration)

