#!/usr/local/bin/python3
import matplotlib.pyplot as plt
import numpy as np


# PLOT NORMS
cgIN = open('cg_sor.txt' , 'r')
richardsonIN = open('richardson_sor.txt', 'r')


cgIt = []; cgNorm = []

for line in cgIN.readlines():

    cgIt.append( float( line.split()[0]))
    cgNorm.append( float( line.split()[4] ) )

richIt = []; richNorm = []

for line in richardsonIN.readlines():

    richIt.append( float( line.split()[0] ) )
    richNorm.append( float( line.split()[4] ) )


plt.clf()

plt.plot( cgIt , cgNorm , 'r-', label='Conjugate Gradients')
plt.plot( richIt, richNorm, 'k-', label='Richarson')
plt.xlabel( 'Iteration' ), plt.ylabel( 'Norm Residual' )
plt.yscale('log')
plt.legend( loc='upper right' , numpoints = 1 )
plt.title( 'Residual Convergence Comparison')

plt.savefig( 'CGRichPlot.pdf' )





# PLOT EXACT Solutions
x = np.linspace(0,1,1000)
y = np.linspace(0,1,1000)
z = np.linspace(0,1,1000)


X, Y = np.meshgrid(x, y)
Z = 1 + X**2 + 2*Y**2

plt.clf()
plt.figure(figsize=(12,10),dpi=200)
plt.contourf( X, Y, Z , 100, cmap='coolwarm')
plt.colorbar(ticks=[1.0,1.5,2.0,2.5,3,3.5,4.0]);
plt.xlabel( 'x' ), plt.ylabel( 'y' )
plt.title( '$u(x,y) = 1 + x^2 +2y^2$')
plt.savefig("Quad_Exact.png")

Z = np.sin(2*np.pi*X) + np.sin(2*np.pi*Y)

plt.clf()
plt.figure(figsize=(12,10),dpi=200)
plt.contourf( X, Y, Z , 100, cmap='coolwarm')
plt.colorbar(ticks=[-2.0,-1.0,0.0,1.0,2.0]);
plt.xlabel( 'x' ), plt.ylabel( 'y' )
plt.title( '$u(x,y) = sin(2 \pi x) + sin(2 \pi y)$')
plt.savefig("Trig_Exact.png")
