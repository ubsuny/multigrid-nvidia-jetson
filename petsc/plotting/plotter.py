#!/usr/local/bin/python3
import matplotlib.pyplot as plt
import numpy as np


# PLOT NORMS
cgIN = open('../helmholtz/output/ex1.out', 'r')
richardsonIN = open('../helmholtz/output/ex2.out', 'r')
mg_VIN = open('../helmholtz/output/ex3.out', 'r')
mg_FIN = open('../helmholtz/output/ex4.out', 'r')

cg_trigIN = open('../helmholtz/output/ex1_trig.out', 'r')
richardson_trigIN = open('../helmholtz/output/ex2_trig.out', 'r')
mg_V_trigIN = open('../helmholtz/output/ex3_trig.out', 'r')
mg_F_trigIN = open('../helmholtz/output/ex4_trig.out', 'r')


cgIt   = []; cgRes   = []; cgErr   = []
richIt = []; richRes = []; richErr = []
mgVIt  = []; mgVRes  = []; mgVErr  = []
mgFIt  = []; mgFRes  = []; mgFErr  = []

cg_trigIt   = []; cg_trigRes   = []; cg_trigErr   = []
rich_trigIt = []; rich_trigRes = []; rich_trigErr = []
mgV_trigIt  = []; mgV_trigRes  = []; mgV_trigErr  = []
mgF_trigIt  = []; mgF_trigRes  = []; mgF_trigErr  = []

for line in cgIN.readlines():

	cgIt.append( float( line.split()[0]))
	cgErr.append( float( line.split()[4] ) )
	cgRes.append( float( line.split()[7] ) )

for line in richardsonIN.readlines():

	richIt.append( float( line.split()[0] ) )
	richErr.append( float( line.split()[4] ) )
	richRes.append( float( line.split()[7] ) )

for line in mg_VIN.readlines():

	mgVIt.append( float( line.split()[0]))
	mgVErr.append( float( line.split()[4] ) )
	mgVRes.append( float( line.split()[7] ) )

for line in mg_FIN.readlines():

	mgFIt.append( float( line.split()[0]))
	mgFErr.append( float( line.split()[4] ) )
	mgFRes.append( float( line.split()[7] ) )

for line in cg_trigIN.readlines():

	cg_trigIt.append( float( line.split()[0]))
	cg_trigErr.append( float( line.split()[4] ) )
	cg_trigRes.append( float( line.split()[7] ) )

for line in richardson_trigIN.readlines():

	rich_trigIt.append( float( line.split()[0] ) )
	rich_trigErr.append( float( line.split()[4] ) )
	rich_trigRes.append( float( line.split()[7] ) )

for line in mg_V_trigIN.readlines():

	mgV_trigIt.append( float( line.split()[0]))
	mgV_trigErr.append( float( line.split()[4] ) )
	mgV_trigRes.append( float( line.split()[7] ) )

for line in mg_F_trigIN.readlines():

	mgF_trigIt.append( float( line.split()[0]))
	mgF_trigErr.append( float( line.split()[4] ) )
	mgF_trigRes.append( float( line.split()[7] ) )

#### Full Plots ####
plt.clf()
plt.plot( cgIt  , cgRes , 'r-', label='Conjugate Gradients')
plt.plot( richIt, richRes, 'k-', label='Richarson')
plt.plot( mgVIt,  mgVRes, 'b-', label='Multigrid V-Cycle')
plt.plot( mgFIt,  mgFRes, 'g-', label='Multigrid F-Cycle')
plt.xlabel( 'Iteration' ), plt.ylabel( 'Norm Residual' )
plt.yscale('log')
plt.legend( loc='lower right' , numpoints = 1 )
plt.title( 'Residual Convergence Comparison\nu(x,y) = 1 + x^2 + 2*y^2')
plt.savefig( 'Helmholtz_Residual.pdf' )


plt.clf()
plt.plot( cgIt  , cgErr , 'r-', label='Conjugate Gradients')
plt.plot( richIt, richErr, 'k-', label='Richarson')
plt.plot( mgVIt,  mgVErr, 'b-', label='Multigrid V-Cycle')
plt.plot( mgFIt,  mgFErr, 'g-', label='Multigrid F-Cycle')
plt.xlabel( 'Iteration' ), plt.ylabel( 'Norm Residual' )
plt.yscale('log')
plt.legend( loc='lower right' , numpoints = 1 )
plt.title( 'Residual Convergence Comparison\nu(x,y) = 1 + x^2 + 2*y^2')
plt.savefig( 'Helmholtz_Error.pdf' )


plt.clf()
plt.plot( cg_trigIt  , cg_trigRes , 'r-', label='Conjugate Gradients')
plt.plot( rich_trigIt, rich_trigRes, 'k-', label='Richarson')
plt.plot( mgV_trigIt,  mgV_trigRes, 'b-', label='Multigrid V-Cycle')
plt.plot( mgF_trigIt,  mgF_trigRes, 'g-', label='Multigrid F-Cycle')
plt.xlabel( 'Iteration' ), plt.ylabel( 'Norm Residual' )
plt.yscale('log')
plt.legend( loc='lower right' , numpoints = 1 )
plt.title( 'Residual Convergence Comparison\nu(x,y) = Sin(2*pi*x) + Sin(2*pi*y)')
plt.savefig( 'Helmholtz_Trig_Residual.pdf' )


plt.clf()
plt.plot( cg_trigIt  , cg_trigErr , 'r-', label='Conjugate Gradients')
plt.plot( rich_trigIt, rich_trigErr, 'k-', label='Richarson')
plt.plot( mgV_trigIt,  mgV_trigErr, 'b-', label='Multigrid V-Cycle')
plt.plot( mgF_trigIt,  mgF_trigErr, 'g-', label='Multigrid F-Cycle')
plt.xlabel( 'Iteration' ), plt.ylabel( 'Norm Residual' )
plt.yscale('log')
plt.legend( loc='lower right' , numpoints = 1 )
plt.title( 'Residual Convergence Comparison\nu(x,y) = Sin(2*pi*x) + Sin(2*pi*y)')
plt.savefig( 'Helmholtz_Trig_Error.pdf' )


#### Separate Plots ####
plt.clf()
fig, axs = plt.subplots(2, 2, constrained_layout=True)
axs[0,0].plot(cgIt  , cgRes , 'r-', label='Conjugate Gradients')
axs[0,0].plot( richIt, richRes, 'k-', label='Richarson')
axs[0,0].set_title('Residual Norm KSP Methods', fontsize=10)
axs[0,0].set_xlabel('Iteration', fontsize=6)
axs[0,0].set_ylabel('Norm Residual', fontsize=6)
axs[0,0].tick_params(labelsize=6)
axs[0,0].set_yscale('log')
axs[0,0].legend( loc='lower right' , numpoints = 1, fontsize=6 )

axs[0,1].plot(cgIt  , cgErr , 'r-', label='Conjugate Gradients')
axs[0,1].plot( richIt, richErr, 'k-', label='Richarson')
axs[0,1].set_title('Error Norm KSP Solvers', fontsize=10)
axs[0,1].set_xlabel('Iteration', fontsize=6)
axs[0,1].set_ylabel('Norm Error', fontsize=6)
axs[0,1].tick_params(labelsize=6)
axs[0,1].set_yscale('log')
axs[0,1].legend( loc='lower right' , numpoints = 1, fontsize=6 )

axs[1,0].plot( mgVIt,  mgVRes, 'r-', label='Multigrid V-Cycle')
axs[1,0].plot( mgFIt,  mgFRes, 'k-', label='Multigrid F-Cycle')
axs[1,0].set_title('Residual Norm Multigrid Methods', fontsize=10)
axs[1,0].set_xlabel('Iteration', fontsize=6)
axs[1,0].set_ylabel('Norm Residual', fontsize=6)
axs[1,0].tick_params(labelsize=6)
axs[1,0].set_yscale('log')
axs[1,0].legend( loc='upper right' , numpoints = 1, fontsize=6 )

axs[1,1].plot( mgVIt,  mgVErr, 'r-', label='Multigrid V-Cycle')
axs[1,1].plot( mgFIt,  mgFErr, 'k-', label='Multigrid F-Cycle')
axs[1,1].set_title('Error Norm Multigrid Methods', fontsize=10)
axs[1,1].set_xlabel('Iteration', fontsize=6)
axs[1,1].set_ylabel('Norm Error', fontsize=6)
axs[1,1].tick_params(labelsize=6)
axs[1,1].set_yscale('log')
axs[1,1].legend( loc='upper right' , numpoints = 1, fontsize=6 )
plt.savefig( 'Helmholtz_Subplots.png', dpi=500)


plt.clf()
fig, axs = plt.subplots(2, 2, constrained_layout=True)
axs[0,0].plot(cg_trigIt  , cg_trigRes , 'r-', label='Conjugate Gradients')
axs[0,0].plot( rich_trigIt, rich_trigRes, 'k-', label='Richarson')
axs[0,0].set_title('Residual Norm KSP Methods', fontsize=10)
axs[0,0].set_xlabel('Iteration', fontsize=6)
axs[0,0].set_ylabel('Norm Residual', fontsize=6)
axs[0,0].tick_params(labelsize=6)
axs[0,0].set_yscale('log')
axs[0,0].legend( loc='lower right' , numpoints = 1, fontsize=6 )

axs[0,1].plot(cg_trigIt  , cg_trigErr , 'r-', label='Conjugate Gradients')
axs[0,1].plot( rich_trigIt, rich_trigErr, 'k-', label='Richarson')
axs[0,1].set_title('Error Norm KSP Solvers', fontsize=10)
axs[0,1].set_xlabel('Iteration', fontsize=6)
axs[0,1].set_ylabel('Norm Error', fontsize=6)
axs[0,1].tick_params(labelsize=6)
axs[0,1].set_yscale('log')
axs[0,1].legend( loc='lower right' , numpoints = 1, fontsize=6 )

axs[1,0].plot( mgV_trigIt,  mgV_trigRes, 'r-', label='Multigrid V-Cycle')
axs[1,0].plot( mgF_trigIt,  mgF_trigRes, 'k-', label='Multigrid F-Cycle')
axs[1,0].set_title('Residual Norm Multigrid Methods', fontsize=10)
axs[1,0].set_xlabel('Iteration', fontsize=6)
axs[1,0].set_ylabel('Norm Residual', fontsize=6)
axs[1,0].tick_params(labelsize=6)
axs[1,0].set_yscale('log')
axs[1,0].legend( loc='upper right' , numpoints = 1, fontsize=6 )

axs[1,1].plot( mgV_trigIt,  mgV_trigErr, 'r-', label='Multigrid V-Cycle')
axs[1,1].plot( mgF_trigIt,  mgF_trigErr, 'k-', label='Multigrid F-Cycle')
axs[1,1].set_title('Error Norm Multigrid Methods', fontsize=10)
axs[1,1].set_xlabel('Iteration', fontsize=6)
axs[1,1].set_ylabel('Norm Error', fontsize=6)
axs[1,1].tick_params(labelsize=6)
axs[1,1].set_yscale('log')
axs[1,1].legend( loc='upper right' , numpoints = 1, fontsize=6 )
plt.savefig( 'Helmholtz_Trig_Subplots.png', dpi=500)


#### PLOT EXACT Solutions ####
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
