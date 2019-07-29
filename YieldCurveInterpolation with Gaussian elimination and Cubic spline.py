import numpy as np
import matplotlib.pyplot as plt

x=year=np.array([2,3,4,5,7,10,15])
y=rate=np.array([6.01,6.11,6.16,6.22,6.32,6.43,6.56])
n=len(x)
L=np.diag(np.ones(len(year),float))
R=np.array([0]*n,float)

#build matrix on left-hand side
for i in range(1,n-1):
    for j in range(i,i+1):
        L[i,j]=2*(x[j]-x[j-1]+x[j+1]-x[j])
for i in range(1,n-1):
    for j in range(i,i+1):
        L[i,j-1]=x[j]-x[j-1]
for i in range(1,n-1):
    for j in range(i,i+1):
        L[i,j+1]=x[j+1]-x[j]

#build matrix on right-hand side
for i in range(1,n-1):
    R[i]=3*((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]))
print('L matrix: \n{},\nR matrix: \n{}'.format(L,R))

# Gaussian elimination method to solve linear equation
#input: L matrix and R matrix. Output: solution x matrix
def GaussElimination(L,R):
    #combine L and R matrix
    R=np.array([R]).T
    m=np.column_stack((L,R))
    
    #Elimination matrix
    i=0
    while i<len(m)-1:
        if m[i,i]==0:
            for j in range(len(m)+1):
                m[i,j],m[i+1,j]=m[i+1,j],m[i,j]
        for j in range(i+1,len(m)):
            if m[j,i] ==0 : continue
            factor=m[i,i]/m[j,i]
            m[j]=factor*m[j]
            m[j]=m[i]-m[j]
        i+=1
    a=m[:len(m),:len(m)]
    b=m[:,-1]
    
    #Back-substitition
    n=len(m)
    x=np.zeros(len(m),float)
    x[n-1]=b[n-1]/a[n-1,n-1]     #compute 1st x 
    for i in range(n-2,-1,-1):
        term=0
        for j in range(i+1,n):
            term+=a[i,j]*x[j]
        x[i]=(b[i]-term)/a[i,i]
    
    return x

#solution to c
c=GaussElimination(L,R) 

d=np.array([0]*(len(c)-1),float)
b=np.array([0]*(len(c)-1),float)
a=np.array([0]*(len(c)-1),float)

for i in range(1,len(x)):
    d[i-1]=(c[i]-c[i-1])/(3*(x[i]-x[i-1]))
for i in range(1,len(x)):
    b[i-1]=(y[i]-y[i-1])/(x[i]-x[i-1])-(x[i]-x[i-1])*(2*c[i-1]+c[i])/3
for i in range(1,len(y)):
    a[i-1]=y[i-1]
print('Coefficients a,b,c,d: \na: {} \nb: {} \nc: {} \nd: {}'.format(a,b.round(4),c.round(4),d.round(4)))

f0=lambda p : a[0]+b[0]*(p-x[0])+c[0]*(p-x[0])**2+d[0]*(p-x[0])**3
f1=lambda p : a[1]+b[1]*(p-x[1])+c[1]*(p-x[1])**2+d[1]*(p-x[1])**3
f2=lambda p : a[2]+b[2]*(p-x[2])+c[2]*(p-x[2])**2+d[2]*(p-x[2])**3
f3=lambda p : a[3]+b[3]*(p-x[3])+c[3]*(p-x[3])**2+d[3]*(p-x[3])**3
f4=lambda p : a[4]+b[4]*(p-x[4])+c[4]*(p-x[4])**2+d[4]*(p-x[4])**3
f5=lambda p : a[5]+b[5]*(p-x[5])+c[5]*(p-x[5])**2+d[5]*(p-x[5])**3

p1=np.arange(x[0],x[1],0.0001)
p2=np.arange(x[1],x[2],0.0001)
p3=np.arange(x[2],x[3],0.0001)
p4=np.arange(x[3],x[4],0.0001)
p5=np.arange(x[4],x[5],0.0001)
p6=np.arange(x[5],x[6],0.0001)

#plot yield curve
plt.figure(figsize=(14,10))
plt.scatter(x,y,marker='o',lw=2,label='Exact swap rate')
plt.plot(p1,f0(p1),'b-',label='Interval [{},{}]: {:.4f}+{:.4f}*(x-{:})+{:.4f}*(x-{:})^2+{:.4f}*(x-{:})^3'.format(
    x[0],x[1],a[0],b[0],x[0],c[0],x[0],d[0],x[0]))
plt.plot(p2,f1(p2),'y--',label='Interval [{},{}]: {:.4f}+{:.4f}*(x-{:})+{:.4f}*(x-{:})^2+{:.4f}*(x-{:})^3'.format(
    x[1],x[2],a[1],b[1],x[1],c[1],x[1],d[1],x[1]))
plt.plot(p3,f2(p3),'k-',label='Interval [{},{}]: {:.4f}+{:.4f}*(x-{:})+{:.4f}*(x-{:})^2+{:.4f}*(x-{:})^3'.format(
    x[2],x[3],a[2],b[2],x[2],c[2],x[2],d[2],x[2]))
plt.plot(p4,f3(p4),'c-.',label='Interval [{},{}]: {:.4f}+{:.4f}*(x-{:})+{:.4f}*(x-{:})^2+{:.4f}*(x-{:})^3'.format(
    x[3],x[4],a[3],b[3],x[3],c[3],x[3],d[3],x[3]))
plt.plot(p5,f4(p5),'m:',label='Interval [{},{}]: {:.4f}+{:.4f}*(x-{:})+{:.4f}*(x-{:})^2+{:.4f}*(x-{:})^3'.format(
    x[4],x[5],a[4],b[4],x[4],c[4],x[4],d[4],x[4]))
plt.plot(p6,f5(p6),'g-',label='Interval [{},{}]: {:.4f}+{:.4f}*(x-{:})+{:.4f}*(x-{:})^2+{:.4f}*(x-{:})^3'.format(
    x[5],x[6],a[5],b[5],x[5],c[5],x[5],d[5],x[5]))
plt.xlabel('years')
plt.ylabel('Swap rate %')
plt.grid()
plt.legend()
plt.show()
