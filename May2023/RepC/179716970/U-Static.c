#include "Formulations/Displacements/U-Newmark-beta.h"
#include "Macros.h"
#include "Types.h"
#include "petscsnes.h"
#include <petscistypes.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct {
Mask ActiveNodes;
Mask ActiveDOFs;
IS Dirichlet_dofs;
Particle MPM_Mesh;
Mesh FEM_Mesh;
Vec Lumped_Mass;
} Ctx;
static int *__create_sparsity_pattern(Mask ActiveNodes, Particle MPM_Mesh);
static IS __get_dirichlet_list_dofs(Mask ActiveNodes, Mesh FEM_Mesh, int Step,
int NumTimeStep);
static PetscErrorCode __compute_nodal_lumped_mass(Vec Lumped_MassMatrix,
Particle MPM_Mesh,
Mesh FEM_Mesh,
Mask ActiveNodes);
static PetscErrorCode __form_initial_guess(Vec DU, Mesh FEM_Mesh, Mask ActiveNodes);
static PetscErrorCode __lagrangian_evaluation(SNES snes, Vec D_U, Vec Residual,
void *ctx);
static PetscErrorCode __local_compatibility_conditions(const PetscScalar *dU,
Mask ActiveNodes,
Particle MPM_Mesh,
Mesh FEM_Mesh);
static PetscErrorCode __constitutive_update(Particle MPM_Mesh, Mesh FEM_Mesh);
static PetscErrorCode __nodal_internal_forces(PetscScalar *Lagrangian,
Mask ActiveNodes, Mask ActiveDOFs,
Particle MPM_Mesh, Mesh FEM_Mesh);
static void __nodal_traction_forces(PetscScalar *Lagrangian, Mask ActiveNodes,
Mask ActiveDOFs, Particle MPM_Mesh,
Mesh FEM_Mesh);
static void
__nodal_inertial_forces(PetscScalar *Lagrangian, const PetscScalar *M_II,
Mask ActiveNodes,
Mask ActiveDOFs);
static PetscErrorCode __jacobian_evaluation(SNES snes, Vec dU, Mat Jacobian,
Mat B, void *ctx);
static PetscErrorCode __monitor(PetscInt Time, PetscInt NumTimeStep,
PetscInt SNES_Iter, PetscInt KSP_Iter,
PetscInt SNES_MaxIter, PetscScalar KSP_Norm,
PetscScalar SNES_Norm,
SNESConvergedReason converged_reason);
static PetscErrorCode __update_Particles(Vec dU,
Particle MPM_Mesh, Mesh FEM_Mesh,
Mask ActiveNodes);
static unsigned InitialStep;
static unsigned NumTimeStep;
static unsigned TimeStep;
PetscErrorCode U_Static(Mesh FEM_Mesh, Particle MPM_Mesh,
Time_Int_Params Parameters_Solver) {
PetscErrorCode STATUS = EXIT_SUCCESS;
unsigned Ndim = NumberDimensions;
unsigned Nactivenodes;
unsigned Ntotaldofs;
InitialStep = Parameters_Solver.InitialTimeStep;
NumTimeStep = Parameters_Solver.NumTimeStep;
TimeStep = InitialStep;
double DeltaTimeStep;
double DeltaX = FEM_Mesh.DeltaX;
Mat Tangent_Stiffness;
int *sparsity_pattern;
Vec Lumped_Mass;
Vec Residual;
Vec DU;
IS Dirichlet_dofs;
Mask ActiveNodes;
Mask ActiveDOFs;
SNES snes;
KSP ksp;
PC pc;
Ctx AplicationCtx;
unsigned SNES_Max_Iter = Parameters_Solver.MaxIter;
double Relative_TOL = Parameters_Solver.TOL_Newmark_beta;
double Absolute_TOL = 100 * Relative_TOL;
if (Driver_EigenErosion) {
compute_Beps__Constitutive__(MPM_Mesh, FEM_Mesh, true);
}
while (TimeStep < NumTimeStep) {
DoProgress("Simulation:", TimeStep, NumTimeStep);
STATUS = local_search__MeshTools__(MPM_Mesh, FEM_Mesh);
if (STATUS == EXIT_FAILURE) {
fprintf(stderr, "" RED " Error in " RESET "" BOLDRED
"local_search__MeshTools__() " RESET " \n");
return EXIT_FAILURE;
}
ActiveNodes = get_active_nodes__MeshTools__(FEM_Mesh);
Nactivenodes = ActiveNodes.Nactivenodes;
Ntotaldofs = Ndim * Nactivenodes;
ActiveDOFs = get_active_dofs__MeshTools__(ActiveNodes, FEM_Mesh, TimeStep,
NumTimeStep);
sparsity_pattern = __create_sparsity_pattern(ActiveNodes, MPM_Mesh);
if ((Driver_EigenErosion == true) || (Driver_EigenSoftening == true)) {
compute_Beps__Constitutive__(MPM_Mesh, FEM_Mesh, false);
}
VecCreate(PETSC_COMM_WORLD, &Lumped_Mass);
VecSetSizes(Lumped_Mass, PETSC_DECIDE, Ntotaldofs);
VecSetFromOptions(Lumped_Mass);
PetscCall(__compute_nodal_lumped_mass(Lumped_Mass, MPM_Mesh, FEM_Mesh,
ActiveNodes));
VecAssemblyBegin(Lumped_Mass);
VecAssemblyEnd(Lumped_Mass);
Dirichlet_dofs =
__get_dirichlet_list_dofs(ActiveNodes, FEM_Mesh, TimeStep, NumTimeStep);
AplicationCtx.ActiveNodes = ActiveNodes;
AplicationCtx.ActiveDOFs = ActiveDOFs;
AplicationCtx.Dirichlet_dofs = Dirichlet_dofs;
AplicationCtx.MPM_Mesh = MPM_Mesh;
AplicationCtx.FEM_Mesh = FEM_Mesh;
AplicationCtx.Lumped_Mass = Lumped_Mass;
PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
PetscCall(SNESSetType(snes, SNESNEWTONLS));
PetscCall(SNESSetOptionsPrefix(snes, "SolidLagragian_"));
PetscCall(VecCreate(PETSC_COMM_WORLD, &DU));
PetscCall(VecSetSizes(DU, PETSC_DECIDE, Ntotaldofs));
PetscCall(VecSetFromOptions(DU));
PetscCall(VecSetOption(DU, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
PetscCall(VecDuplicate(DU, &Residual));
PetscCall(VecSetFromOptions(Residual));
PetscCall(VecSetOption(Residual, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, Ntotaldofs, Ntotaldofs, 0,
sparsity_pattern, &Tangent_Stiffness));
PetscCall(
MatSetOption(Tangent_Stiffness, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
PetscCall(MatSetFromOptions(Tangent_Stiffness));
PetscCall(SNESSetFunction(snes, Residual, __lagrangian_evaluation,
&AplicationCtx));
PetscCall(SNESSetJacobian(snes, Tangent_Stiffness, Tangent_Stiffness,
__jacobian_evaluation, &AplicationCtx));
Petsc_Direct_solver = true;
Petsc_Iterative_solver = false;
if(Petsc_Direct_solver)
{
PetscCall(SNESGetKSP(snes, &ksp));
PetscCall(KSPGetPC(ksp, &pc));
PetscCall(PCSetType(pc, PCCHOLESKY));
PCFactorSetMatSolverType(pc,MATSOLVERCHOLMOD);
PetscCall(SNESSetTolerances(snes, Absolute_TOL, Relative_TOL, PETSC_DEFAULT,
SNES_Max_Iter, PETSC_DEFAULT));
PetscCall(KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT,
PETSC_DEFAULT));
}
else if(Petsc_Iterative_solver)
{
PetscCall(SNESGetKSP(snes, &ksp));
PetscCall(KSPGetPC(ksp, &pc));
PetscCall(PCSetType(pc, PCJACOBI));
PetscCall(SNESSetTolerances(snes, Absolute_TOL, Relative_TOL, PETSC_DEFAULT,
SNES_Max_Iter, PETSC_DEFAULT));
PetscCall(KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT,
PETSC_DEFAULT));
}
PetscCall(SNESSetLagJacobian(snes, 1));
PetscCall(SNESSetFromOptions(snes));
PetscCall(VecZeroEntries(DU));
__form_initial_guess(DU, FEM_Mesh, ActiveNodes);    
PetscCall(SNESSolve(snes, PETSC_NULL, DU));
if (Flag_Print_Convergence) {
SNESConvergedReason converged_reason;
PetscInt SNES_Iter, KSP_Iter;
PetscScalar KSP_Norm, SNES_Norm;
VecNorm(Residual, NORM_2, &SNES_Norm);
KSPGetResidualNorm(ksp, &KSP_Norm);
SNESGetConvergedReason(snes, &converged_reason);
SNESGetIterationNumber(snes, &SNES_Iter);
SNESGetLinearSolveIterations(snes, &KSP_Iter);
__monitor(TimeStep, NumTimeStep, SNES_Iter, KSP_Iter, SNES_Max_Iter,
KSP_Norm, SNES_Norm, converged_reason);
}
PetscCall(MatDestroy(&Tangent_Stiffness));
PetscCall(VecDestroy(&Residual));
PetscCall(SNESDestroy(&snes));
PetscCall(
__update_Particles(DU, MPM_Mesh, FEM_Mesh, ActiveNodes));
if (TimeStep % ResultsTimeStep == 0) {
particle_results_vtk__InOutFun__(MPM_Mesh, TimeStep, ResultsTimeStep);
}
PetscCall(VecDestroy(&Lumped_Mass));
PetscCall(VecDestroy(&DU));
PetscCall(ISDestroy(&Dirichlet_dofs));
free(ActiveNodes.Nodes2Mask);
free(ActiveDOFs.Nodes2Mask);
free(sparsity_pattern);
TimeStep++;
}
return EXIT_SUCCESS;
}
static IS __get_dirichlet_list_dofs(Mask ActiveNodes, Mesh FEM_Mesh, int Step,
int NumTimeStep) {
int Ndim = NumberDimensions;
int Nnodes_mask = ActiveNodes.Nactivenodes;
int Order = Nnodes_mask * Ndim;
int Order_dirichlet = 0;
int Number_of_BCC = FEM_Mesh.Bounds.NumBounds;
int NumNodesBound; 
int NumDimBound;   
int Id_BCC;        
int Id_BCC_mask;
int Id_BCC_mask_k;
PetscInt *List_active_dofs;
PetscCalloc1(Order, &List_active_dofs);
for (int i = 0; i < Number_of_BCC; i++) {
NumNodesBound = FEM_Mesh.Bounds.BCC_i[i].NumNodes;
NumDimBound = FEM_Mesh.Bounds.BCC_i[i].Dim;
for (int j = 0; j < NumNodesBound; j++) {
Id_BCC = FEM_Mesh.Bounds.BCC_i[i].Nodes[j];
Id_BCC_mask = ActiveNodes.Nodes2Mask[Id_BCC];
if (Id_BCC_mask != -1) {
for (int k = 0; k < NumDimBound; k++) {
Id_BCC_mask_k = Id_BCC_mask * Ndim + k;
if ((FEM_Mesh.Bounds.BCC_i[i].Dir[k * NumTimeStep + Step] == 1) && 
(List_active_dofs[Id_BCC_mask_k] == 0)) {
List_active_dofs[Id_BCC_mask_k] = 1;
Order_dirichlet++;
}
}
}
}
}
PetscInt *List_active_dirchlet_dofs;
PetscInt aux_idx = 0;
PetscMalloc1(Order_dirichlet, &List_active_dirchlet_dofs);
for (int A_i = 0; A_i < Order; A_i++) {
if (List_active_dofs[A_i] == 1) {
List_active_dirchlet_dofs[aux_idx] = A_i;
aux_idx++;
}
}
IS Dirichlet_dofs;
ISCreateGeneral(PETSC_COMM_WORLD, Order_dirichlet, List_active_dirchlet_dofs,
PETSC_USE_POINTER, &Dirichlet_dofs);
PetscFree(List_active_dofs);
return Dirichlet_dofs;
}
static PetscErrorCode __form_initial_guess(Vec DU, Mesh FEM_Mesh, Mask ActiveNodes) {
unsigned Ndim = NumberDimensions;
unsigned Nactivenodes = ActiveNodes.Nactivenodes;
PetscInt Ntotaldofs = Ndim * Nactivenodes;
PetscInt Dof_Ai;
PetscScalar *DU_ptr;
PetscCall(VecGetArray(DU, &DU_ptr));
unsigned NumBounds = FEM_Mesh.Bounds.NumBounds;
for (unsigned i = 0; i < NumBounds; i++) {
unsigned NumNodesBound = FEM_Mesh.Bounds.BCC_i[i].NumNodes;
for (unsigned j = 0; j < NumNodesBound; j++) {
int Id_BCC = FEM_Mesh.Bounds.BCC_i[i].Nodes[j];
int Id_BCC_mask = ActiveNodes.Nodes2Mask[Id_BCC];
if (Id_BCC_mask == -1) {
continue;
}
for (unsigned k = 0; k < Ndim; k++) {
if (FEM_Mesh.Bounds.BCC_i[i].Dir[k * NumTimeStep + TimeStep] == 1) {
Dof_Ai = Id_BCC_mask * Ndim + k;
DU_ptr[Dof_Ai] = FEM_Mesh.Bounds.BCC_i[i].Value[k].Fx[TimeStep];
}
}
}
}
PetscCall(VecRestoreArray(DU, &DU_ptr));
return EXIT_SUCCESS;
}
static PetscErrorCode __lagrangian_evaluation(SNES snes, Vec dU, Vec Lagrangian,
void *ctx) {
PetscErrorCode STATUS = EXIT_SUCCESS;
Mask ActiveNodes = ((Ctx *)ctx)->ActiveNodes;
Mask ActiveDOFs = ((Ctx *)ctx)->ActiveDOFs;
Particle MPM_Mesh = ((Ctx *)ctx)->MPM_Mesh;
Mesh FEM_Mesh = ((Ctx *)ctx)->FEM_Mesh;
Vec Lumped_Mass = ((Ctx *)ctx)->Lumped_Mass;  
unsigned Ndim = NumberDimensions;
unsigned Nactivenodes = ActiveNodes.Nactivenodes;
unsigned Ntotaldofs = Ndim * Nactivenodes;
PetscScalar *Lagrangian_ptr;
const PetscScalar *dU_ptr;
const PetscScalar *Lumped_Mass_ptr;
PetscCall(VecZeroEntries(Lagrangian));
PetscCall(VecGetArray(Lagrangian, &Lagrangian_ptr));
PetscCall(VecGetArrayRead(Lumped_Mass, &Lumped_Mass_ptr));
PetscCall(VecGetArrayRead(dU, &dU_ptr));
PetscCall(__local_compatibility_conditions(dU_ptr, ActiveNodes,
MPM_Mesh, FEM_Mesh));
PetscCall(__constitutive_update(MPM_Mesh, FEM_Mesh));
PetscCall(__nodal_internal_forces(Lagrangian_ptr, ActiveNodes, ActiveDOFs,
MPM_Mesh, FEM_Mesh));
__nodal_traction_forces(Lagrangian_ptr, ActiveNodes, ActiveDOFs, MPM_Mesh,
FEM_Mesh);
__nodal_inertial_forces(Lagrangian_ptr, Lumped_Mass_ptr, ActiveNodes, ActiveDOFs);
PetscCall(VecRestoreArray(Lagrangian, &Lagrangian_ptr));
PetscCall(VecRestoreArrayRead(dU, &dU_ptr));
return STATUS;
}
static PetscErrorCode __local_compatibility_conditions(const PetscScalar *dU,
Mask ActiveNodes,
Particle MPM_Mesh,
Mesh FEM_Mesh) {
PetscErrorCode STATUS = EXIT_SUCCESS;
unsigned Ndim = NumberDimensions;
unsigned Np = MPM_Mesh.NumGP;
unsigned NumberNodes_p;
unsigned Order_p;
unsigned p;
int Idx_Element_p;
int Idx_Patch_p;
#pragma omp parallel private(NumberNodes_p, Order_p)
{
#pragma omp for private(p)
for (p = 0; p < Np; p++) {
NumberNodes_p = MPM_Mesh.NumberNodes[p];
Element Nodes_p =
nodal_set__Particles__(p, MPM_Mesh.ListNodes[p], NumberNodes_p);
Order_p = NumberNodes_p * Ndim;
double *D_Displacement_Ap = (double *)calloc(Order_p, __SIZEOF_DOUBLE__);
if (D_Displacement_Ap == NULL) {
fprintf(stderr, "" RED "Error in calloc(): Out of memory" RESET " \n");
STATUS = EXIT_FAILURE;
}
get_set_field__MeshTools__(D_Displacement_Ap, dU, Nodes_p, ActiveNodes);
Matrix gradient_p = compute_dN__MeshTools__(Nodes_p, MPM_Mesh, FEM_Mesh);
double *F_n_p = MPM_Mesh.Phi.F_n.nM[p];
double *F_n1_p = MPM_Mesh.Phi.F_n1.nM[p];
double *DF_p = MPM_Mesh.Phi.DF.nM[p];
update_increment_Deformation_Gradient__Particles__(
DF_p, D_Displacement_Ap, gradient_p.nV, NumberNodes_p);
update_Deformation_Gradient_n1__Particles__(F_n1_p, F_n_p, DF_p);
MPM_Mesh.Phi.J_n1.nV[p] = I3__TensorLib__(F_n1_p);
if (MPM_Mesh.Phi.J_n1.nV[p] <= 0.0) {
fprintf(stderr,
"" RED "Negative jacobian in particle %i: %e" RESET " \n", p,
MPM_Mesh.Phi.J_n1.nV[p]);
STATUS = EXIT_FAILURE;
}
if (FEM_Mesh.Locking_Control_Fbar) {
Idx_Element_p = MPM_Mesh.Element_p[p];
Idx_Patch_p = FEM_Mesh.Idx_Patch[Idx_Element_p];
FEM_Mesh.Vol_Patch_n[Idx_Patch_p] +=
MPM_Mesh.Phi.J_n.nV[p] * MPM_Mesh.Phi.Vol_0.nV[p];
FEM_Mesh.Vol_Patch_n1[Idx_Patch_p] +=
MPM_Mesh.Phi.J_n1.nV[p] * MPM_Mesh.Phi.Vol_0.nV[p];
}
free(D_Displacement_Ap);
free__MatrixLib__(gradient_p);
free(Nodes_p.Connectivity);
}
#pragma omp barrier
if (FEM_Mesh.Locking_Control_Fbar) {
double Vn_patch;
double Vn1_patch;
double J_patch;
#pragma omp for private(p, Vn_patch, Vn1_patch, J_patch, Idx_Element_p,        Idx_Patch_p)
for (p = 0; p < Np; p++) {
Idx_Element_p = MPM_Mesh.Element_p[p];
Idx_Patch_p = FEM_Mesh.Idx_Patch[Idx_Element_p];
Vn_patch = FEM_Mesh.Vol_Patch_n[Idx_Patch_p];
Vn1_patch = FEM_Mesh.Vol_Patch_n1[Idx_Patch_p];
J_patch = Vn1_patch / Vn_patch;
STATUS = get_locking_free_Deformation_Gradient_n1__Particles__(
p, J_patch, MPM_Mesh);
if (STATUS == EXIT_FAILURE) {
fprintf(
stderr,
"" RED "Error in "
"get_locking_free_Deformation_Gradient_n1__Particles__()" RESET
" \n");
STATUS = EXIT_FAILURE;
}
MPM_Mesh.Phi.Jbar.nV[p] *= J_patch;
}
}
}
return STATUS;
}
static PetscErrorCode __constitutive_update(Particle MPM_Mesh, Mesh FEM_Mesh) {
PetscErrorCode STATUS = EXIT_SUCCESS;
int STATUS_p = EXIT_SUCCESS;
unsigned Np = MPM_Mesh.NumGP;
unsigned MatIndx_p;
unsigned p;
#pragma omp for private(p, MatIndx_p, STATUS_p)
for (p = 0; p < Np; p++) {
if ((Driver_EigenErosion == true) || (Driver_EigenSoftening == true)) {
if (MPM_Mesh.Phi.Damage_n[p] == 1.0) {
MPM_Mesh.Phi.W[p] = 0.0;
continue;
}
}
MatIndx_p = MPM_Mesh.MatIdx[p];
Material MatProp_p = MPM_Mesh.Mat[MatIndx_p];
STATUS_p = Stress_integration__Constitutive__(p, MPM_Mesh, MatProp_p);
if (STATUS_p == EXIT_FAILURE) {
fprintf(stderr,
"" RED "Error in Stress_integration__Constitutive__(%i,,)" RESET
" \n",
p);
STATUS = STATUS_p;
}
}
return STATUS;
}
static PetscErrorCode __nodal_internal_forces(PetscScalar *Lagrangian,
Mask ActiveNodes, Mask ActiveDOFs,
Particle MPM_Mesh,
Mesh FEM_Mesh) {
PetscErrorCode STATUS = EXIT_SUCCESS;
unsigned Ndim = NumberDimensions;
unsigned Np = MPM_Mesh.NumGP;
unsigned NumNodes_p;
unsigned p;
#if NumberDimensions == 2
double InternalForcesDensity_Ap[2];
int Mask_dofs_A[2];
#else
double InternalForcesDensity_Ap[3];
int Mask_dofs_A[3];
#endif
double *Damage_field_n1 = MPM_Mesh.Phi.Damage_n1;
const double *Vol_0 = MPM_Mesh.Phi.Vol_0.nV;
#pragma omp parallel private(NumNodes_p, InternalForcesDensity_Ap, Mask_dofs_A)
{
#pragma omp for private(p)
for (p = 0; p < Np; p++) {
PetscErrorCode STATUS_p;
double V0_p = Vol_0[p];
double *DF_p = MPM_Mesh.Phi.DF.nM[p];
NumNodes_p = MPM_Mesh.NumberNodes[p];
Element Nodes_p =
nodal_set__Particles__(p, MPM_Mesh.ListNodes[p], NumNodes_p);
Matrix d_shapefunction_n_p =
compute_dN__MeshTools__(Nodes_p, MPM_Mesh, FEM_Mesh);
double *d_shapefunction_n1_p = push_forward_dN__MeshTools__(
d_shapefunction_n_p.nV, DF_p, NumNodes_p, &STATUS_p);
if (STATUS_p == EXIT_FAILURE) {
fprintf(stderr, "" RED " Error in " RESET "" BOLDRED
"push_forward_dN__MeshTools__() " RESET " \n");
STATUS = EXIT_FAILURE;
}
double *kirchhoff_p = MPM_Mesh.Phi.Stress.nM[p];
if ((Driver_EigenErosion == true) || (Driver_EigenSoftening == true)) {
STATUS_p = compute_damage__Constitutive__(p, MPM_Mesh, FEM_Mesh.DeltaX);
if (STATUS_p == EXIT_FAILURE) {
fprintf(stderr, "" RED " Error in " RESET "" BOLDRED
"compute_damage__Constitutive__() " RESET " \n");
STATUS = EXIT_FAILURE;
}
#if NumberDimensions == 2
for (unsigned i = 0; i < 5; i++) {
kirchhoff_p[i] *= (1.0 - Damage_field_n1[p]);
}
#else
for (unsigned i = 0; i < 9; i++) {
kirchhoff_p[i] *= (1.0 - Damage_field_n1[p]);
}
#endif
}
for (unsigned A = 0; A < NumNodes_p; A++) {
double *d_shapefunction_n1_pA = &d_shapefunction_n1_p[A * Ndim];
int Ap = Nodes_p.Connectivity[A];
int Mask_node_A = ActiveNodes.Nodes2Mask[Ap];
for (unsigned i = 0; i < Ndim; i++) {
InternalForcesDensity_Ap[i] = 0.0;
for (unsigned j = 0; j < Ndim; j++) {
InternalForcesDensity_Ap[i] +=
kirchhoff_p[i * Ndim + j] * d_shapefunction_n1_pA[j];
}
Mask_dofs_A[i] = Mask_node_A * Ndim + i;
}
#pragma omp critical
{
for (unsigned i = 0; i < Ndim; i++) {
if (ActiveDOFs.Nodes2Mask[Mask_dofs_A[i]] != -1) {
Lagrangian[Mask_dofs_A[i]] += InternalForcesDensity_Ap[i] * V0_p;
}
}
} 
}   
free__MatrixLib__(d_shapefunction_n_p);
free(d_shapefunction_n1_p);
free(Nodes_p.Connectivity);
} 
}   
return STATUS;
}
static void __nodal_traction_forces(PetscScalar *Lagrangian, Mask ActiveNodes,
Mask ActiveDOFs, Particle MPM_Mesh,
Mesh FEM_Mesh) {
unsigned Ndim = NumberDimensions;
unsigned NumContactForces = MPM_Mesh.Neumann_Contours.NumBounds;
unsigned NumNodesLoad;
Load Load_i;
Element Nodes_p; 
Matrix N_p;      
double N_pa;
double A0_p;
#if NumberDimensions == 2
double T[2] = {0.0, 0.0};
#else
double T[3] = {0.0, 0.0};
#endif
#if NumberDimensions == 2
double LocalTractionForce_Ap[2];
int Mask_dofs_A[2];
#else
double LocalTractionForce_Ap[3];
int Mask_dofs_A[3];
#endif
unsigned p;
int Ap, Mask_node_A;
unsigned NumNodes_p; 
for (unsigned cf_idx = 0; cf_idx < NumContactForces; cf_idx++) {
Load_i = MPM_Mesh.Neumann_Contours.BCC_i[cf_idx];
NumNodesLoad = Load_i.NumNodes;
for (unsigned nl_idx = 0; nl_idx < NumNodesLoad; nl_idx++) {
p = Load_i.Nodes[nl_idx];
#if NumberDimensions == 2
A0_p = MPM_Mesh.Phi.Vol_0.nV[p] / Thickness_Plain_Stress;
#else
A0_p = MPM_Mesh.Phi.Area_0.nV[p];
#endif
NumNodes_p = MPM_Mesh.NumberNodes[p];
Nodes_p = nodal_set__Particles__(p, MPM_Mesh.ListNodes[p], NumNodes_p);
N_p = compute_N__MeshTools__(Nodes_p, MPM_Mesh, FEM_Mesh);
for (unsigned i = 0; i < Ndim; i++) {
if (Load_i.Dir[i * NumTimeStep + TimeStep] == 1) {
T[i] = Load_i.Value[i].Fx[TimeStep];
}
}
for (unsigned A = 0; A < NumNodes_p; A++) {
N_pa = N_p.nV[A];
Ap = Nodes_p.Connectivity[A];
Mask_node_A = ActiveNodes.Nodes2Mask[Ap];
for (unsigned i = 0; i < Ndim; i++) {
LocalTractionForce_Ap[i] = -N_pa * T[i];
Mask_dofs_A[i] = Mask_node_A * Ndim + i;
}
#pragma omp critical
{
for (unsigned i = 0; i < Ndim; i++) {
if (ActiveDOFs.Nodes2Mask[Mask_dofs_A[i]] != -1) {
Lagrangian[Mask_dofs_A[i]] += LocalTractionForce_Ap[i] * A0_p;
}
}
} 
} 
free__MatrixLib__(N_p);
free(Nodes_p.Connectivity);
}
}
}
static void
__nodal_inertial_forces(PetscScalar *Lagrangian, const PetscScalar *M_II,
Mask ActiveNodes,
Mask ActiveDOFs) {
unsigned Ndim = NumberDimensions;
unsigned Nactivenodes = ActiveNodes.Nactivenodes;
unsigned Ntotaldofs = Ndim * Nactivenodes;
unsigned idx;
#if NumberDimensions == 2
double b[2] = {0.0, 0.0};
#else
double b[3] = {0.0, 0.0, 0.0};
#endif
if (gravity_field.STATUS == true) {
for (unsigned i = 0; i < Ndim; i++) {
b[i] = gravity_field.Value[i].Fx[TimeStep];
}
}
#pragma omp for private(idx)
for (idx = 0; idx < Ntotaldofs; idx++) {
#pragma omp critical
{
if (ActiveDOFs.Nodes2Mask[idx] != -1) {
Lagrangian[idx] +=
- M_II[idx] * b[idx % Ndim];
}
}
}
}
static int *__create_sparsity_pattern(Mask ActiveNodes, Particle MPM_Mesh) {
unsigned Ndim = NumberDimensions;
unsigned Ntotaldofs = Ndim * ActiveNodes.Nactivenodes;
unsigned Np = MPM_Mesh.NumGP;
unsigned NumNodes_p;
int *Active_dof_Mat = (int *)calloc(Ntotaldofs * Ntotaldofs, __SIZEOF_INT__);
Element Nodes_p;
int Ap, Mask_node_A, Dof_Ai;
int Bp, Mask_node_B, Dof_Bj;
for (unsigned p = 0; p < Np; p++) {
NumNodes_p = MPM_Mesh.NumberNodes[p];
Nodes_p = nodal_set__Particles__(p, MPM_Mesh.ListNodes[p], NumNodes_p);
for (unsigned A = 0; A < NumNodes_p; A++) {
Ap = Nodes_p.Connectivity[A];
Mask_node_A = ActiveNodes.Nodes2Mask[Ap];
for (unsigned B = 0; B < NumNodes_p; B++) {
Bp = Nodes_p.Connectivity[B];
Mask_node_B = ActiveNodes.Nodes2Mask[Bp];
for (unsigned i = 0; i < Ndim; i++) {
Dof_Ai = Mask_node_A * Ndim + i;
for (unsigned j = 0; j < Ndim; j++) {
Dof_Bj = Mask_node_B * Ndim + j;
Active_dof_Mat[Dof_Ai * Ntotaldofs + Dof_Bj] = 1;
}
}
}
}
free(Nodes_p.Connectivity);
}
int *sparsity_pattern = (int *)calloc(Ntotaldofs, __SIZEOF_INT__);
for (unsigned A = 0; A < Ntotaldofs; A++) {
for (unsigned B = 0; B < Ntotaldofs; B++) {
sparsity_pattern[A] += Active_dof_Mat[A * Ntotaldofs + B];
}
}
free(Active_dof_Mat);
return sparsity_pattern;
}
static PetscErrorCode __compute_nodal_lumped_mass(Vec Lumped_MassMatrix,
Particle MPM_Mesh,
Mesh FEM_Mesh,
Mask ActiveNodes) {
PetscErrorCode STATUS = EXIT_SUCCESS;
unsigned Ndim = NumberDimensions;
unsigned Np = MPM_Mesh.NumGP;
unsigned NumberNodes_p;
unsigned p;
#if NumberDimensions == 2
double Local_Mass_Matrix_p[2];
int Mask_dofs_A[2];
#else
double Local_Mass_Matrix_p[3];
int Mask_dofs_A[3];
#endif
#pragma omp parallel private(NumberNodes_p, Local_Mass_Matrix_p, Mask_dofs_A)
{
#pragma omp for private(p)
for (p = 0; p < Np; p++) {
NumberNodes_p = MPM_Mesh.NumberNodes[p];
Element Nodes_p =
nodal_set__Particles__(p, MPM_Mesh.ListNodes[p], NumberNodes_p);
Matrix ShapeFunction_p =
compute_N__MeshTools__(Nodes_p, MPM_Mesh, FEM_Mesh);
double m_p = MPM_Mesh.Phi.mass.nV[p];
for (unsigned A = 0; A < NumberNodes_p; A++) {
int Ap = Nodes_p.Connectivity[A];
int Mask_node_A = ActiveNodes.Nodes2Mask[Ap];
double Na_p = ShapeFunction_p.nV[A];
double M_AB_p = Na_p * m_p;
for (unsigned i = 0; i < Ndim; i++) {
Local_Mass_Matrix_p[i] = M_AB_p;
Mask_dofs_A[i] = Mask_node_A * Ndim + i;
}
#pragma omp critical
{
VecSetValues(Lumped_MassMatrix, Ndim, Mask_dofs_A,
Local_Mass_Matrix_p, ADD_VALUES);
} 
}   
free__MatrixLib__(ShapeFunction_p);
free(Nodes_p.Connectivity);
} 
}   
return STATUS;
}
static PetscErrorCode __jacobian_evaluation(SNES snes, Vec dU, Mat Jacobian,
Mat Preconditioner, void *ctx) {
Mask ActiveNodes = ((Ctx *)ctx)->ActiveNodes;
IS Dirichlet_dofs = ((Ctx *)ctx)->Dirichlet_dofs;
Particle MPM_Mesh = ((Ctx *)ctx)->MPM_Mesh;
Mesh FEM_Mesh = ((Ctx *)ctx)->FEM_Mesh;
PetscErrorCode STATUS = EXIT_SUCCESS;
unsigned Ndim = NumberDimensions;
unsigned Np = MPM_Mesh.NumGP;
unsigned NumNodes_p;
unsigned MatIndx_p;
unsigned p;
#if NumberDimensions == 2
double Jacobian_p[4];
double Stiffness_density_p[4];
int Mask_dofs_A[2];
int Mask_dofs_B[2];
#else
double Jacobian_p[9];
double Stiffness_density_p[9];
int Mask_dofs_A[3];
int Mask_dofs_B[3];
#endif
PetscCall(MatZeroEntries(Jacobian));
#pragma omp parallel private(NumNodes_p, MatIndx_p, Stiffness_density_p,       Jacobian_p, Mask_dofs_A, Mask_dofs_B)
{
#pragma omp for private(p)
for (p = 0; p < Np; p++) {
double m_p = MPM_Mesh.Phi.mass.nV[p];
double V0_p = MPM_Mesh.Phi.Vol_0.nV[p];
MatIndx_p = MPM_Mesh.MatIdx[p];
Material MatProp_p = MPM_Mesh.Mat[MatIndx_p];
double *DF_p = MPM_Mesh.Phi.DF.nM[p];
NumNodes_p = MPM_Mesh.NumberNodes[p];
Element Nodes_p =
nodal_set__Particles__(p, MPM_Mesh.ListNodes[p], NumNodes_p);
Matrix shapefunction_n_p =
compute_N__MeshTools__(Nodes_p, MPM_Mesh, FEM_Mesh);
Matrix d_shapefunction_n_p =
compute_dN__MeshTools__(Nodes_p, MPM_Mesh, FEM_Mesh);
double *d_shapefunction_n1_p = push_forward_dN__MeshTools__(
d_shapefunction_n_p.nV, DF_p, NumNodes_p, &STATUS);
if (STATUS == EXIT_FAILURE) {
fprintf(stderr,
"" RED "Error in push_forward_dN__MeshTools__()" RESET " \n");
STATUS = EXIT_FAILURE;
}
for (unsigned A = 0; A < NumNodes_p; A++) {
double shapefunction_n_pA = shapefunction_n_p.nV[A];
double *d_shapefunction_n_pA = d_shapefunction_n_p.nM[A];
int Ap = Nodes_p.Connectivity[A];
int Mask_node_A = ActiveNodes.Nodes2Mask[Ap];
for (unsigned B = 0; B < NumNodes_p; B++) {
double shapefunction_n_pB = shapefunction_n_p.nV[B];
double *d_shapefunction_n_pB = d_shapefunction_n_p.nM[B];
int Bp = Nodes_p.Connectivity[B];
int Mask_node_B = ActiveNodes.Nodes2Mask[Bp];
STATUS = stiffness_density__Constitutive__(
p, Stiffness_density_p, &d_shapefunction_n1_p[A * Ndim],
&d_shapefunction_n1_p[B * Ndim], d_shapefunction_n_pA,
d_shapefunction_n_pB, 0.0, MPM_Mesh, MatProp_p);
if (STATUS == EXIT_FAILURE) {
fprintf(stderr,
"" RED "Error in stiffness_density__Constitutive__" RESET
"\n");
}
if ((Driver_EigenErosion == true) ||
(Driver_EigenSoftening == true)) {
double damage_p = MPM_Mesh.Phi.Damage_n1[p];
for (unsigned i = 0; i < Ndim * Ndim; i++) {
Stiffness_density_p[i] *= (1.0 - damage_p);
}
}
for (unsigned i = 0; i < Ndim; i++) {
for (unsigned j = 0; j < Ndim; j++) {
Jacobian_p[i * Ndim + j] =
Stiffness_density_p[i * Ndim + j] * V0_p;
}
Mask_dofs_A[i] = Mask_node_A * Ndim + i;
Mask_dofs_B[i] = Mask_node_B * Ndim + i;
}
#pragma omp critical
{
MatSetValues(Jacobian, Ndim, Mask_dofs_A, Ndim, Mask_dofs_B,
Jacobian_p, ADD_VALUES);
} 
}   
}     
free__MatrixLib__(shapefunction_n_p);
free__MatrixLib__(d_shapefunction_n_p);
free(d_shapefunction_n1_p);
free(Nodes_p.Connectivity);
} 
}   
PetscCall(MatAssemblyBegin(Jacobian, MAT_FINAL_ASSEMBLY));
PetscCall(MatAssemblyEnd(Jacobian, MAT_FINAL_ASSEMBLY));
PetscCall(MatZeroRowsColumnsIS(Jacobian, Dirichlet_dofs, 1.0, NULL, NULL));
return STATUS;
}
static PetscErrorCode __update_Particles(Vec dU, 
Particle MPM_Mesh, Mesh FEM_Mesh,
Mask ActiveNodes) {
PetscErrorCode STATUS = EXIT_SUCCESS;
unsigned Ndim = NumberDimensions;
unsigned Np = MPM_Mesh.NumGP;
unsigned NumNodes_p;
unsigned p;
double DU_pI;
const PetscScalar *dU_ptr;
PetscCall(VecGetArrayRead(dU, &dU_ptr));
#pragma omp parallel private(NumNodes_p, DU_pI)
{
#pragma omp for private(p)
for (p = 0; p < Np; p++) {
MPM_Mesh.Phi.J_n.nV[p] = MPM_Mesh.Phi.J_n1.nV[p];
MPM_Mesh.Phi.rho.nV[p] =
MPM_Mesh.Phi.mass.nV[p] /
(MPM_Mesh.Phi.Vol_0.nV[p] * MPM_Mesh.Phi.J_n.nV[p]);
MPM_Mesh.Phi.Kappa_n[p] = MPM_Mesh.Phi.Kappa_n1[p];
MPM_Mesh.Phi.EPS_n[p] = MPM_Mesh.Phi.EPS_n1[p];
#if NumberDimensions == 2
for (unsigned i = 0; i < 5; i++)
MPM_Mesh.Phi.b_e_n.nM[p][i] = MPM_Mesh.Phi.b_e_n1.nM[p][i];
#else
for (unsigned i = 0; i < 9; i++)
MPM_Mesh.Phi.b_e_n.nM[p][i] = MPM_Mesh.Phi.b_e_n1.nM[p][i];
#endif
if ((Driver_EigenErosion == true) || (Driver_EigenSoftening == true)) {
MPM_Mesh.Phi.Damage_n[p] = MPM_Mesh.Phi.Damage_n1[p];
}
if (Driver_EigenSoftening == true) {
MPM_Mesh.Phi.Strain_f_n[p] = MPM_Mesh.Phi.Strain_f_n1[p];
}
#if NumberDimensions == 2
for (unsigned i = 0; i < 5; i++)
MPM_Mesh.Phi.F_n.nM[p][i] = MPM_Mesh.Phi.F_n1.nM[p][i];
#else
for (unsigned i = 0; i < 9; i++)
MPM_Mesh.Phi.F_n.nM[p][i] = MPM_Mesh.Phi.F_n1.nM[p][i];
#endif
NumNodes_p = MPM_Mesh.NumberNodes[p];
Element Nodes_p =
nodal_set__Particles__(p, MPM_Mesh.ListNodes[p], NumNodes_p);
Matrix ShapeFunction_p =
compute_N__MeshTools__(Nodes_p, MPM_Mesh, FEM_Mesh);
for (unsigned A = 0; A < NumNodes_p; A++) {
double ShapeFunction_pI = ShapeFunction_p.nV[A];
int Ap = Nodes_p.Connectivity[A];
int A_mask = ActiveNodes.Nodes2Mask[Ap];
for (unsigned i = 0; i < Ndim; i++) {
DU_pI = ShapeFunction_pI * dU_ptr[A_mask * Ndim + i];
MPM_Mesh.Phi.dis.nM[p][i] += DU_pI;
MPM_Mesh.Phi.x_GC.nM[p][i] += DU_pI;
}
} 
free(Nodes_p.Connectivity);
free__MatrixLib__(ShapeFunction_p);
} 
}   
PetscCall(VecRestoreArrayRead(dU, &dU_ptr));
return STATUS;
}
static PetscErrorCode __monitor(PetscInt Time, PetscInt NumTimeStep,
PetscInt SNES_Iter, PetscInt KSP_Iter,
PetscInt SNES_MaxIter, PetscScalar KSP_Norm,
PetscScalar SNES_Norm,
SNESConvergedReason converged_reason) {
if (NumTimeStep < 10) {
PetscPrintf(PETSC_COMM_WORLD, "" GREEN "Step" RESET ": [%01d/%01d] | ",
Time, NumTimeStep);
} else if (NumTimeStep < 100) {
PetscPrintf(PETSC_COMM_WORLD, "" GREEN "Step" RESET ": [%02d/%02d] | ",
Time, NumTimeStep);
} else if (NumTimeStep < 1000) {
PetscPrintf(PETSC_COMM_WORLD, "" GREEN "Step" RESET ": [%03d/%03d] | ",
Time, NumTimeStep);
} else if (NumTimeStep < 10000) {
PetscPrintf(PETSC_COMM_WORLD, "" GREEN "Step" RESET ": [%04d/%04d] | ",
Time, NumTimeStep);
} else if (NumTimeStep < 100000) {
PetscPrintf(PETSC_COMM_WORLD, "" GREEN "Step" RESET ": [%05d/%05d] | ",
Time, NumTimeStep);
} else if (NumTimeStep < 1000000) {
PetscPrintf(PETSC_COMM_WORLD, "" GREEN "Step" RESET ": [%i/%i] | ", Time,
NumTimeStep);
}
PetscPrintf(PETSC_COMM_WORLD, "" GREEN "SNES L2-norm" RESET ": %1.4e | ",
SNES_Norm);
if (SNES_MaxIter < 10) {
PetscPrintf(PETSC_COMM_WORLD,
"" GREEN "SNES Iterations" RESET ": [%01d/%01d] | ", SNES_Iter,
SNES_MaxIter);
} else if (SNES_MaxIter < 100) {
PetscPrintf(PETSC_COMM_WORLD,
"" GREEN "SNES Iterations" RESET ": [%02d/%02d] | ", SNES_Iter,
SNES_MaxIter);
}
PetscPrintf(PETSC_COMM_WORLD, "" GREEN "KSP L2-norm" RESET ": %1.4e | ",
KSP_Norm);
PetscPrintf(PETSC_COMM_WORLD, "" GREEN "KSP Iterations" RESET ": %02d | ",
KSP_Iter);
PetscPrintf(PETSC_COMM_WORLD, "" GREEN "Converged reason" RESET ": %s \n",
SNESConvergedReasons[converged_reason]);
FILE *Stats_Solver;
char Name_file_t[10000];
sprintf(Name_file_t, "%s/Stats_Solver.csv", OutputDir);
Stats_Solver = fopen(Name_file_t, "a");
if (Time == 0) {
fprintf(Stats_Solver, "%s,%s,%s,%s,%s\n", "SNES Iterations",
"KSP Iterations", "SNES L2-norm", "KSP L2-norm",
"Converged reason");
}
fprintf(Stats_Solver, "%i,%i,%1.4e,%1.4e,%s\n", SNES_Iter, KSP_Iter,
SNES_Norm, KSP_Norm, SNESConvergedReasons[converged_reason]);
fclose(Stats_Solver);
return EXIT_SUCCESS;
}
