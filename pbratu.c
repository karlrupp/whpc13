static const char help[] = "p-Bratu nonlinear PDE in 2d.\n\
We solve the  p-Laplacian (nonlinear diffusion) combined with\n\
the Bratu (solid fuel ignition) nonlinearity in a 2D rectangular\n\
domain, using distributed arrays (DAs) to partition the parallel grid.\n\
The command line options include:\n\
  -p <2>: `p' in p-Laplacian term\n\
  -epsilon <1e-05>: Strain-regularization in p-Laplacian\n\
  -lambda <6>: Bratu parameter\n\
\n";


/* ------------------------------------------------------------------------

    p-Laplacian and Solid Fuel Ignition problem.  This problem is modeled by
    the partial differential equation

        -div(eta grad(u)) - lambda exp(u) = 0,  0 < x,y < 1,

    with closure

        eta(gamma) = (epsilon^2 + gamma)^((p-2)/2),   gamma = 1/2 |grad u|^2

    with boundary conditions

        u = 0  for  x = 0, x = 1, y = 0, y = 1.

    A finite difference approximation a 9-point stencil is used to discretize
    the boundary value problem to obtain a nonlinear system of equations.
    This would be a 5-point stencil if not for the p-Laplacian's nonlinearity.

    Program usage:  mpiexec -n <procs> ./pbratu [-help] [all PETSc options]
     e.g.,
      ./pbratu -fd_jacobian -mat_fd_coloring_view_draw -draw_pause -1
      mpiexec -n 2 ./pbratu -fd_jacobian_ghosted -log_summary

  ------------------------------------------------------------------------- */

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h    - linear solvers        petscdm.h  - generic grid/physics management
*/
#include "petscdmda.h"
#include "petscsnes.h"

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct {
  PetscReal lambda;         /* Bratu parameter */
  PetscReal p;              /* Exponent in p-Laplacian */
  PetscReal epsilon;        /* Regularization */
} AppCtx;

/*
   User-defined routines
*/
static PetscErrorCode FormInitialGuess(DM,Vec);
static PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar**,PetscScalar**,AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES                   snes;                 /* nonlinear solver */
  Vec                    x,r;                  /* solution, residual vectors */
  AppCtx                 user;                 /* user-defined work context */
  DM                     dm;
  PetscInt               its;                  /* iterations for convergence */
  SNESConvergedReason    reason;               /* Check convergence */
  PetscReal              bratu_lambda_max = 42.0,bratu_lambda_min = 0.;
  PetscErrorCode         ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscInitialize(&argc,&argv,0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.lambda = 6.0; user.p = 2.0; user.epsilon = 1e-5;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"p-Bratu options",__FILE__);CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-lambda","Bratu parameter","",user.lambda,&user.lambda,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-p","Exponent `p' in p-Laplacian","",user.p,&user.p,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-epsilon","Strain-regularization in p-Laplacian","",user.epsilon,&user.epsilon,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (user.lambda > bratu_lambda_max || user.lambda < bratu_lambda_min) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING: lambda %g out of range for p=2\n",user.lambda);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                      -4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&dm);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Attach the DM to SNES, used for coarsening, refinement, and callbacks
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DM; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(dm,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set local residual evaluation routine
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDASNESSetFunctionLocal(dm,INSERT_VALUES,(DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm,&user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */

  ierr = FormInitialGuess(dm,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s Number of Newton iterations = %D\n",SNESConvergedReasons[reason],its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
static PetscErrorCode FormInitialGuess(DM dm,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscScalar    **x;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dm,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DMDAVecGetArray(dm,X,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)

  */
  ierr = DMDAGetCorners(dm,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        /* boundary conditions are all zero Dirichlet */
        x[j][i] = 0.0;
      } else {
          const PetscReal
            xx = 2*(PetscReal)i/(Mx-1) - 1,
            yy = 2*(PetscReal)j/(My-1) - 1;
          x[j][i] = (1 - xx*xx) * (1-yy*yy);
      }
    }
  }

  /*
     Restore vector
  */
  ierr = DMDAVecRestoreArray(dm,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/*
   FormFunctionLocal - Evaluates nonlinear function, F(x).
 */
static PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscScalar **x,PetscScalar **f,AppCtx *user)
{
  PetscReal      hx,hy,hxdhy,hydhx,sc;
  PetscInt       i,j;

  PetscFunctionBegin;
  hx    = 1./(PetscReal)(info->mx-1);
  hy    = 1./(PetscReal)(info->my-1);
  hxdhy = hx/hy;
  hydhx = hy/hx;
  sc    = hx*hy*user->lambda;
  /*
     Compute function over the locally owned part of the grid
  */
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        f[j][i] = x[j][i];      /* homogeneous Dirichlet boundary condition */
      } else {
        const PetscScalar
          u = x[j][i],
          uxx = (2.0*u - x[j][i-1] - x[j][i+1])*hydhx,
          uyy = (2.0*u - x[j-1][i] - x[j+1][i])*hxdhy;
        f[j][i] = uxx + uyy - sc*PetscExpScalar(u);
      }
    }
  }
  PetscFunctionReturn(0);
}
