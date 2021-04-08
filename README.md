# GPU/CPU Multigrid Comparison for the "Good" Helmholtz for NVIDIA Jetson

### Literature Review:

A detailed review of the topics explored in this project can be found in the docs folder.


### Implementation:

##### -> petsc/helmholtz/helmholtz.c
 Implementation of the multigrid solver for the "Good" helmholtz equation using PETSc.  This code is designed for CPUs.

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
  - Views solution if -potential_view is given in command line











[PETSc](https://www.mcs.anl.gov/petsc/index.html) - PETSc is a suite of data structures and routines for the scalable (parallel) solution of scientific applications modeled by partial differential equations. It supports MPI, and GPUs through CUDA or OpenCL, as well as hybrid MPI-GPU parallelism.

Useful Resources:
[Dave May Extreme Scale Multigrid Components Within PETSc](https://arxiv.org/pdf/1604.07163.pdf)
