# GPU/CPU Multigrid Comparison for the "Good" Helmholtz for NVIDIA Jetson

### Literature Review:

A detailed review of the topics explored in this project as well as results from 8 test runs can be found in the docs folder.


### Implementation:




##### -> petsc/helmholtz/helmholtz.c
Implementation of the multigrid solver for the "Good" helmholtz equation using PETSc.  This code is designed for CPUs.  To run the helmholtz solver, PETSc must be installed and configured.  Instructions for PETSc's installation can be found [here](https://www.mcs.anl.gov/petsc/documentation/installation.html). The configure options used on the NVIDIA Jetson were:
 
 - --download-f2cblaslapack,
 - --download-hdf5,
 - --download-openmpi,
 - --download-triangle,
 - --with-cc=gcc,
 - --with-cxx=g++,
 - --with-debugging,
 - --with-fc=0,
 - --with-valgrind,
 - PETSC_ARCH=arch-linux-c-debug

The Makefile can be run with just the command *make*.  Command line options can be passed to the code to specify the solver types and other solver options.  8 example runs are provided at the bottom of the helmholtz code.  The example runs output the error and residual norms from the Krylov solver as well as an HDF5 format file containing the potential solution.

###### How it works
There are two structure that are PETSc uses throughout the code:
- *DM* objects are used to manage communication between the algebraic structures in PETSc (Vec and Mat) and mesh data structures in PDE-based (or other) simulations. See, for example, DMDACreate().
- *PetscDS* is a PETSc object that manages a discrete system, which is a set of discretizations + continuum equations from a PetscWeakForm


###### Subroutines -
- *ProcessOptions* - Processes command line options and sets any constants.
- *CreateMesh* - Creates unstructured mesh using the PETSc's [DMPLEX structure](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DMPLEX/DMPLEX.html)
- *SetupPrimalProblem* - This subroutine is where the weak formulation is defined for the helmholtz equation.  The key functions are [PetscDSSetResidual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DT/PetscDSSetResidual.html) and [PetscDSSetJacobian](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DT/PetscDSSetJacobian.html)
- *SetupDiscretization* - This subroutine creates the finite element discretization.
- main - In order this function:
  - Initializes the Petsc code.
  - Creates nonlinear solver (SNES) object
  - Runs mesh creation subroutine
  - Runs option processing subroutine
  - Creates DM object
  - Runs discretization subroutine
  - Creates solution vector *u*
  - Initializes u to 0
  - Creates name "potential" for solution vector
  - Use DMPlex's internal FEM routines to compute SNES boundary values, residual, and Jacobian.
  - Sets SNES object options such as solver type.
  - Checks the residual and Jacobian functions using the exact solution by outputting some diagnostic information
  - Runs nonlinear solve step
  - Pulls solve output into u
  - Views solution if -potential\_view is given in command line


##### -> petsc/plotting/plotter.py

The plotter uses files stored in *petsc/helmholtz/output/* to make residual and error plots.  The files in *petsc/helmholtz/output/* files contain the output from -ksp\_monitor\_error.


##### -> petsc/gpu_helmholtz/helmholtz_gpu.cu

This is an incomplete GPU implementation of the helmholtz code.  The PETSc configuration given above has CUDA capabilities but the code is not ready for GPU execution.  

An example of a CUDA enabled PETSc code can be found in petsc/src/snes/tutorials/ex47cu.cu. 


##### Visualizing Solutions

Mentioned above, the example tests included in the helmholtz code output HDF5 files containing the final potentials calculated.  PETSc contains a python script that can convert these *.h5* files into *.xmf* files which can be visualized using a program such as Paraview.  To convert the HDF5 files, run:

$ python PATH\_TO\_PETSC/petsc/lib/petsc/bin/petsc\_gen\_xdmf.py sol\_ex1.h5 


#### Resources

[PETSc](https://www.mcs.anl.gov/petsc/index.html) - PETSc is a suite of data structures and routines for the scalable (parallel) solution of scientific applications modeled by partial differential equations. It supports MPI, and GPUs through CUDA or OpenCL, as well as hybrid MPI-GPU parallelism.

