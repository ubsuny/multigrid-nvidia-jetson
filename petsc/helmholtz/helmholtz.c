static char help[] = "Good Helmholtz Problem in 2d and 3d with finite elements.\n\
We solve the Good Helmholtz problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports automatic convergence estimation\n\
and coarse space adaptivity.\n\n\n";

/*
   The model problem:
      Solve Helmholtz equation on the unit square: (0,1) x (0,1)
          -delta u + u = f,
           where delta = Laplace operator
      Dirichlet b.c.'s on all sides

*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>


typedef struct {
  /* Domain and mesh definition */
  PetscBool trig; /* Use trig function as exact solution */
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 0.0;
  return 0;
}

static void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g0[d*dim+d] = 1.0;
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

/*For Primal Problem*/
static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static PetscErrorCode quad_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 1.0;
  for (d = 0; d < dim; ++d) *u += (d+1)*PetscSqr(x[d]);
  return 0;
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  f0[0] += u[0];
  for (d = 0; d < dim; ++d) f0[0] -= 4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]) + PetscSinReal(2.0*PETSC_PI*x[d]);
}

static void f0_quad_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
	PetscInt d;
	switch (dim) {
		case 1:
	  		f0[0] = 1.0;
			break;
		case 2:
	  		f0[0] = 5.0;
			break;
		case 3:
	  		f0[0] = 13.0;
			break;
		default:
			f0[0] = 5.0;
			break;
  	}
	f0[0] += u[0];
	for (d = 0; d < dim; ++d) f0[0] -= (d+1)*PetscSqr(x[d]);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static PetscErrorCode ProcessOptions(DM dm, AppCtx *options)
{
  MPI_Comm       comm;
  PetscInt	 	 dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  options->trig = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Helmholtz Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-exact_trig", "Use trigonometric exact solution (better for more complex finite elements)", "ex26.c", options->trig, &options->trig, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();


  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create box mesh */
  ierr = DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  /* TODO: This should be pulled into the library */
  {
    char      convType[256];
    PetscBool flg;

    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex12",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;

      ierr = DMConvert(*dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmConv;
      }
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);

  if (user->trig) {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Trig Exact Solution \n");CHKERRQ(ierr);
	ierr = PetscDSSetResidual(ds, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(ds, 0, 0, g0_uu, NULL, NULL, g3_uu);CHKERRQ(ierr);
	ierr = PetscDSSetExactSolution(ds, 0, trig_u, user);CHKERRQ(ierr);
	ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void)) trig_u, NULL, 1, &id, user);CHKERRQ(ierr);
  } else {
	ierr = PetscDSSetResidual(ds, 0, f0_quad_u, f1_u);CHKERRQ(ierr);
	ierr = PetscDSSetJacobian(ds, 0, 0, g0_uu, NULL, NULL, g3_uu);CHKERRQ(ierr);
	ierr = PetscDSSetExactSolution(ds, 0, quad_u, user);CHKERRQ(ierr);
	ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void)) quad_u, NULL, 1, &id, user);CHKERRQ(ierr);
 }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDM(DM dm, PetscFE feAux, AppCtx *user)
{
  DM             dmAux, coordDM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
  ierr = DMClone(dm, &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
  ierr = DMSetField(dmAux, 0, NULL, (PetscObject) feAux);CHKERRQ(ierr);
  ierr = DMCreateDS(dmAux);CHKERRQ(ierr);
  ierr = SetupMaterial(dm, dmAux, user);CHKERRQ(ierr);
  ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  /* Create finite element */
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setup)(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm,cdm);CHKERRQ(ierr);
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
	DM             dm;   /* Problem specification */
	PetscDS        ds;
	SNES           snes; /* Nonlinear solver */
	Vec            u;    /* Solutions */
	AppCtx         user; /* User-defined work context */
	PetscErrorCode ierr;

	ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
	/* Primal system */
	ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
	ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
	ierr = ProcessOptions(dm, &user);CHKERRQ(ierr);
	ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
	ierr = SetupDiscretization(dm, "potential", SetupPrimalProblem, &user);CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
	ierr = VecSet(u, 0.0);CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);
	ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
	ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
	ierr = DMSNESCheckFromOptions(snes, u);CHKERRQ(ierr);

	/*Looking for field error*/
	PetscInt Nfields;
	ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
	ierr = PetscDSGetNumFields(ds, &Nfields);CHKERRQ(ierr);
	ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
	ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
	ierr = VecViewFromOptions(u, NULL, "-potential_view");CHKERRQ(ierr);


	/* Cleanup */
	ierr = VecDestroy(&u);CHKERRQ(ierr);
	ierr = SNESDestroy(&snes);CHKERRQ(ierr);
	ierr = DMDestroy(&dm);CHKERRQ(ierr);
	ierr = PetscFinalize();
	return ierr;
}

/*TEST
test:
  # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 2d_p1_conv
  requires: triangle
  args: -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 3
test:
  # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 2d_p2_conv
  requires: triangle
  args: -potential_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 2
test:
  # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 2d_p3_conv
  requires: triangle
  args: -potential_petscspace_degree 3 -snes_convergence_estimate -convest_num_refine 2
test:
  # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 2d_q1_trig_conv
  args: -exact_trig -dm_plex_box_simplex 0 -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 2
test:
  # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 2d_q2_trig_conv
  args: -exact_trig -dm_plex_box_simplex 0 -potential_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 2
test:
  # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 2d_q3_trig_conv
  args: -exact_trig -dm_plex_box_simplex 0 -potential_petscspace_degree 3 -snes_convergence_estimate -convest_num_refine 2
test:
  # Using -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 3d_p1_conv
  requires: ctetgen
  args: -dm_plex_box_dim 3 -dm_refine 1 -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1
test:
  # Using -dm_refine 1 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 3d_p2_conv
  requires: ctetgen
  args: -dm_plex_box_dim 3 -dm_plex_box_faces 2,2,2 -potential_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 1
test:
  # Using -dm_refine 1 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 3d_p3_conv
  requires: ctetgen
  args: -dm_plex_box_dim 3 -dm_plex_box_faces 2,2,2 -potential_petscspace_degree 3 -snes_convergence_estimate -convest_num_refine 1
test:
  # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 3d_q1_trig_conv
  args: -exact_trig -dm_plex_box_dim 3 -dm_plex_box_simplex 0 -dm_refine 1 -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1
test:
  # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 3d_q2_trig_conv
  args: -exact_trig -dm_plex_box_dim 3 -dm_plex_box_simplex 0 -potential_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 1
test:
  # Using -dm_refine 1 -convest_num_refine 3 we get L_2 convergence rate:
  suffix: 3d_q3_trig_conv
  args: -exact_trig -dm_plex_box_dim 3 -dm_plex_box_simplex 0 -potential_petscspace_degree 3 -snes_convergence_estimate -convest_num_refine 1
test:
  suffix: 2d_p1_gmg_vcycle
  requires: triangle
  args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
    -ksp_type cg -ksp_rtol 1e-10 -pc_type mg \
    -mg_levels_ksp_max_it 1 \
    -mg_levels_pc_type jacobi
test:
  suffix: 2d_p1_gmg_fcycle
  requires: triangle
  args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
    -ksp_type cg -ksp_rtol 1e-10 -pc_type mg -pc_mg_type full \
    -mg_levels_ksp_max_it 2 \
    -mg_levels_pc_type jacobi
test:
  suffix: 2d_p1_gmg_vcycle_trig
  requires: triangle
  args: -exact_trig -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
    -ksp_type cg -ksp_rtol 1e-10 -pc_type mg \
    -mg_levels_ksp_max_it 1 \
    -mg_levels_pc_type jacobi
TEST*/
