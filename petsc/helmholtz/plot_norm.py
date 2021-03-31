import matplotlib.pyplot as plt
import numpy as np

cgIN = open('cg_sor.txt' , 'r')
richardsonIN = open('richardson.txt', 'r')


cgIt = []; cgNorm = []

for line in cgIN.readlines():

    cgIt.append( float( line.split()[0]))
    cgNorm.append( float( line.split()[3] ) )

richIt = []; richNorm = []

for line in richarsonIN.readlines():

    richIt.append( float( line.split()[0] ) )
    richNorm.append( float( line.split()[1] ) )


plt.clf()

plt.plot( cgIt , cgNorm , 'r-', label='Conjugate Gradients')
plt.plot( richIt, richNorm, 'k-', label='Richarson')
plt.xlabel( 'Iteration' ), plt.ylabel( 'Norm' )
plt.yscale('log')
plt.legend( loc='upper right' , numpoints = 1 )
plt.title( 'Residual Convergence Comparison')

plt.savefig( 'CGRichPlot.pdf' )
