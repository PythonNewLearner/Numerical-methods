#define interest rate function
def IR(pmt,year,total,rate):
    n=year*12
    return total-pmt*((1+rate/12)**n-1)/(rate/12)


#Try Secant method
year=20
pmt=400
total=400000
x1,x2=0.05,0.3
iteraterlimit=100
counter=0
print('Iteration \t xnew \t\t x2 \t\t x1')
for iteration in range(iteraterlimit):
    counter+=1
    xnew=x2-(IR(pmt,year,total,x2)*(x2-x1)/(IR(pmt,year,total,x2)-IR(pmt,year,total,x1)))
    if abs(xnew-x2)<1e-5:
        break
    print('{}\t\t {:.6f} \t {:.6f} \t {:.6f}'.format(counter,xnew,x2,x1))
    x1=x2
    x2=xnew
print('Annual interst rate is {:%}'.format(xnew))