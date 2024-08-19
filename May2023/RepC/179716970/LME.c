#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "Macros.h"
#include "Types.h"
#include "Globals.h"
#include "Matlib.h"
#include "Particles.h"
#include "Nodes/LME.h"
double NM_rho_LME = 1.0;
double NM_chi_LME = 2.0;
double NM_gamma_LME = 0.5;
double NM_sigma_LME = 0.5;
double NM_tau_LME = 1E-3;
static double fa__LME__(Matrix, Matrix, double);
static double logZ__LME__(Matrix, Matrix, double);
static Matrix r__LME__(Matrix, Matrix);
static Matrix J__LME__(Matrix, Matrix, Matrix);
static void initialise_lambda__LME__(int, Matrix, Matrix, Matrix, double);
static Matrix gravity_center_Nelder_Mead__LME__(Matrix);
static void order_logZ_simplex_Nelder_Mead__LME__(Matrix, Matrix);
static void expansion_Nelder_Mead__LME__(Matrix, Matrix, Matrix, Matrix, Matrix,
double, double);
static void contraction_Nelder_Mead__LME__(Matrix, Matrix, Matrix, Matrix,
Matrix, double, double);
static void shrinkage_Nelder_Mead__LME__(Matrix, Matrix, Matrix, double);
static ChainPtr tributary__LME__(int, Matrix, double, int, Mesh);
void initialize__LME__(Particle MPM_Mesh, Mesh FEM_Mesh) {
unsigned Ndim = NumberDimensions;
unsigned Np = MPM_Mesh.NumGP; 
unsigned Nelem = FEM_Mesh.NumElemMesh; 
int I0;                                
unsigned p;
bool Is_particle_reachable;
ChainPtr Locality_I0; 
Matrix Delta_Xip;     
Matrix lambda_p;      
double Beta_p;        
#pragma omp parallel shared(Np, Nelem)
{
#pragma omp for private(p, Is_particle_reachable, lambda_p, Beta_p)
for (p = 0; p < Np; p++) {
Is_particle_reachable = false;
unsigned idx_element = 0;
Matrix X_p =
memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_GC.nM[p]);
while ((Is_particle_reachable == false) && (idx_element < Nelem)) {
ChainPtr Elem_p_Connectivity = FEM_Mesh.Connectivity[idx_element];
Matrix Elem_p_Coordinates = get_nodes_coordinates__MeshTools__(
Elem_p_Connectivity, FEM_Mesh.Coordinates);
if (FEM_Mesh.In_Out_Element(X_p, Elem_p_Coordinates) == true) {
Is_particle_reachable = true;
MPM_Mesh.Element_p[p] = idx_element;
MPM_Mesh.I0[p] = get_closest_node__MeshTools__(
X_p, Elem_p_Connectivity, FEM_Mesh.Coordinates);
Beta_p = beta__LME__(gamma_LME, FEM_Mesh.h_avg[MPM_Mesh.I0[p]]);
if (strcmp(wrapper_LME, "Nelder-Mead") == 0) {
initialise_lambda__LME__(p, X_p, Elem_p_Coordinates, lambda_p,
Beta_p);
}
}
free__MatrixLib__(Elem_p_Coordinates);
++idx_element;
}
if (!Is_particle_reachable) {
fprintf(stderr, "%s : %s %i\n", "Error in initialize__LME__()",
"The search algorithm was unable to find particle", p);
exit(EXIT_FAILURE);
}
}
#pragma omp barrier
for (p = 0; p < Np; p++) {
I0 = MPM_Mesh.I0[p];
Locality_I0 = FEM_Mesh.NodalLocality_0[I0];
if ((Driver_EigenErosion == true) || (Driver_EigenSoftening == true)) {
push__SetLib__(&FEM_Mesh.List_Particles_Node[I0], p);
}
while (Locality_I0 != NULL) {
if (FEM_Mesh.ActiveNode[Locality_I0->Idx] == false) {
FEM_Mesh.ActiveNode[Locality_I0->Idx] = true;
}
Locality_I0 = Locality_I0->next;
}
}
#pragma omp for private(p, Delta_Xip, lambda_p, Beta_p)
for (p = 0; p < Np; p++) {
Matrix X_p =
memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_GC.nM[p]);
lambda_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.lambda.nM[p]);
Beta_p = MPM_Mesh.Beta.nV[p];
MPM_Mesh.ListNodes[p] =
tributary__LME__(p, X_p, Beta_p, MPM_Mesh.I0[p], FEM_Mesh);
MPM_Mesh.NumberNodes[p] = lenght__SetLib__(MPM_Mesh.ListNodes[p]);
Delta_Xip = compute_distance__MeshTools__(MPM_Mesh.ListNodes[p], X_p,
FEM_Mesh.Coordinates);
Beta_p = beta__LME__(gamma_LME, FEM_Mesh.h_avg[MPM_Mesh.I0[p]]);
MPM_Mesh.Beta.nV[p] = Beta_p;
__lambda_Newton_Rapson(p, Delta_Xip, lambda_p, Beta_p);
free__MatrixLib__(Delta_Xip);
}
}
}
double beta__LME__(double Gamma, 
double h_avg) 
{
return Gamma / (h_avg * h_avg);
}
static void initialise_lambda__LME__(int Idx_particle, Matrix X_p,
Matrix Elem_p_Coordinates, 
Matrix lambda, 
double Beta)   
{
int Ndim = NumberDimensions;
int Nnodes_simplex = Ndim + 1;
int Size_element = Elem_p_Coordinates.N_rows;
double sqr_dist_i;
int *simplex;
Matrix Norm_l = allocZ__MatrixLib__(Size_element, 1);
Matrix l = allocZ__MatrixLib__(Size_element, Ndim);
Matrix A = allocZ__MatrixLib__(Ndim, Ndim);
Matrix b = allocZ__MatrixLib__(Ndim, 1);
Matrix x;
for (int i = 0; i < Size_element; i++) {
sqr_dist_i = 0.0;
for (int j = 0; j < Ndim; j++) {
l.nM[i][j] = X_p.nV[i] - Elem_p_Coordinates.nM[i][j];
sqr_dist_i += DSQR(l.nM[i][j]);
}
Norm_l.nV[i] = sqr_dist_i;
}
if (Size_element == 3) {
simplex = (int *)Allocate_ArrayZ(Nnodes_simplex, sizeof(int));
simplex[0] = 0;
simplex[1] = 1;
simplex[2] = 2;
} else if (Size_element == 4) {
simplex = (int *)Allocate_ArrayZ(Nnodes_simplex, sizeof(int));
simplex[0] = 0;
simplex[1] = 1;
simplex[2] = 2;
} else {
exit(0);
}
for (int i = 1; i < Nnodes_simplex; i++) {
b.nV[i - 1] = -Beta * (Norm_l.nV[simplex[0]] - Norm_l.nV[simplex[i]]);
for (int j = 0; j < Ndim; j++) {
A.nM[i - 1][j] = l.nM[simplex[i]][j] - l.nM[simplex[0]][j];
}
}
if (rcond__TensorLib__(A.nV) < 1E-8) {
fprintf(stderr, "%s %i : %s \n",
"Error in initialise_lambda__LME__ for particle", Idx_particle,
"The Hessian near to singular matrix!");
exit(EXIT_FAILURE);
}
x = solve__MatrixLib__(A, b);
for (int i = 0; i < Ndim; i++) {
lambda.nV[i] = x.nV[i];
}
free(simplex);
free__MatrixLib__(Norm_l);
free__MatrixLib__(l);
free__MatrixLib__(A);
free__MatrixLib__(b);
free__MatrixLib__(x);
}
static int __lambda_Newton_Rapson(int Idx_particle, Matrix l, Matrix lambda,
double Beta) {
int MaxIter = max_iter_LME;
int Ndim = NumberDimensions;
int NumIter = 0;    
double norm_r = 10; 
Matrix p;           
Matrix r;           
Matrix J;           
Matrix D_lambda;    
while (NumIter <= MaxIter) {
p = p__LME__(l, lambda, Beta);
r = r__LME__(l, p);
norm_r = norm__MatrixLib__(r, 2);
if (norm_r > TOL_wrapper_LME) {
J = J__LME__(l, p, r);
if (rcond__TensorLib__(J.nV) < 1E-8) {
fprintf(stderr,
"" RED "Hessian near to singular matrix: %e" RESET " \n",
rcond__TensorLib__(J.nV));
return EXIT_FAILURE;
}
D_lambda = solve__MatrixLib__(J, r);
for (int i = 0; i < Ndim; i++) {
lambda.nV[i] -= D_lambda.nV[i];
}
free__MatrixLib__(p);
free__MatrixLib__(r);
free__MatrixLib__(J);
free__MatrixLib__(D_lambda);
NumIter++;
} else {
free__MatrixLib__(r);
free__MatrixLib__(p);
break;
}
}
if (NumIter >= MaxIter) {
fprintf(stderr, "%s %i : %s (%i)\n",
"Warning in lambda_Newton_Rapson__LME__ for particle", Idx_particle,
"No convergence reached in the maximum number of interations",
MaxIter);
fprintf(stderr, "%s : %e\n", "Total Error", norm_r);
return EXIT_FAILURE;
}
return EXIT_SUCCESS;
}
void update_lambda_Nelder_Mead__LME__(
int Idx_particle,
Matrix l, 
Matrix lambda, 
double Beta)   
{
int Ndim = NumberDimensions;
int Nnodes_simplex = Ndim + 1;
int MaxIter = 500; 
int NumIter = 0;
Matrix simplex = allocZ__MatrixLib__(Nnodes_simplex, Ndim);
Matrix logZ = allocZ__MatrixLib__(Nnodes_simplex, 1);
Matrix simplex_a = memory_to_matrix__MatrixLib__(1, Ndim, NULL);
Matrix gravity_center;
Matrix reflected_point;
double logZ_reflected_point;
double logZ_0;
double logZ_n;
double logZ_n1;
for (int a = 0; a < Nnodes_simplex; a++) {
for (int i = 0; i < Ndim; i++) {
if (i == a) {
simplex.nM[a][i] = lambda.nV[i] / 10;
} else {
simplex.nM[a][i] = lambda.nV[i];
}
}
}
for (int a = 0; a < Nnodes_simplex; a++) {
simplex_a.nV = simplex.nM[a];
logZ.nV[a] = logZ__LME__(l, simplex_a, Beta);
}
while (NumIter <= MaxIter) {
order_logZ_simplex_Nelder_Mead__LME__(logZ, simplex);
logZ_0 = logZ.nV[0];
logZ_n = logZ.nV[Nnodes_simplex - 2];
logZ_n1 = logZ.nV[Nnodes_simplex - 1];
if (fabs(logZ_0 - logZ_n1) > TOL_wrapper_LME) {
gravity_center = gravity_center_Nelder_Mead__LME__(simplex);
reflected_point = allocZ__MatrixLib__(1, Ndim);
for (int i = 0; i < Ndim; i++) {
reflected_point.nV[i] =
gravity_center.nV[i] +
NM_rho_LME *
(gravity_center.nV[i] - simplex.nM[Nnodes_simplex - 1][i]);
}
logZ_reflected_point = logZ__LME__(l, reflected_point, Beta);
if (logZ_reflected_point < logZ_0) {
expansion_Nelder_Mead__LME__(simplex, logZ, reflected_point,
gravity_center, l, Beta,
logZ_reflected_point);
}
else if ((logZ_reflected_point > logZ_0) &&
(logZ_reflected_point < logZ_n)) {
for (int i = 0; i < Ndim; i++) {
simplex.nM[Nnodes_simplex - 1][i] = reflected_point.nV[i];
}
logZ.nV[Nnodes_simplex - 1] = logZ_reflected_point;
}
else if (logZ_reflected_point >= logZ_n) {
contraction_Nelder_Mead__LME__(simplex, logZ, reflected_point,
gravity_center, l, Beta,
logZ_reflected_point);
}
free__MatrixLib__(reflected_point);
NumIter++;
} else {
break;
}
}
if (NumIter >= MaxIter) {
fprintf(stderr, "%s %i : %s (%i) \n",
"Warning in lambda_Nelder_Mead__LME__ for particle", Idx_particle,
"No convergence reached in the maximum number of interations",
MaxIter);
fprintf(stderr, "%s : %e\n", "Total Error", fabs(logZ_0 - logZ_n1));
}
for (int i = 0; i < Ndim; i++) {
lambda.nV[i] = simplex.nM[0][i];
}
free__MatrixLib__(simplex);
free__MatrixLib__(logZ);
}
static void order_logZ_simplex_Nelder_Mead__LME__(Matrix logZ, Matrix simplex) {
int Ndim = NumberDimensions;
int Nnodes_simplex = Ndim + 1;
bool swapped = false;
double aux;
for (int i = 1; i < Nnodes_simplex; i++) {
swapped = false;
for (int j = 0; j < (Nnodes_simplex - i); j++) {
if (logZ.nV[j] > logZ.nV[j + 1]) {
aux = logZ.nV[j];
logZ.nV[j] = logZ.nV[j + 1];
logZ.nV[j + 1] = aux;
for (int k = 0; k < Ndim; k++) {
aux = simplex.nM[j][k];
simplex.nM[j][k] = simplex.nM[j + 1][k];
simplex.nM[j + 1][k] = aux;
}
swapped = true;
}
}
if (!swapped) {
break;
}
}
}
static Matrix gravity_center_Nelder_Mead__LME__(Matrix simplex) {
int Ndim = NumberDimensions;
int Nnodes_simplex = Ndim + 1;
Matrix gravity_center = allocZ__MatrixLib__(1, Ndim);
for (int i = 0; i < Ndim; i++) {
for (int a = 0; a < Nnodes_simplex; a++) {
gravity_center.nV[i] += simplex.nM[a][i] / Nnodes_simplex;
}
}
return gravity_center;
}
static void expansion_Nelder_Mead__LME__(Matrix simplex, Matrix logZ,
Matrix reflected_point,
Matrix gravity_center, Matrix l,
double Beta,
double logZ_reflected_point) {
int Ndim = NumberDimensions;
int Nnodes_simplex = Ndim + 1;
double logZ_expanded_point;
Matrix expanded_point;
expanded_point = allocZ__MatrixLib__(1, Ndim);
for (int i = 0; i < Ndim; i++) {
expanded_point.nV[i] =
gravity_center.nV[i] +
NM_chi_LME * (reflected_point.nV[i] - gravity_center.nV[i]);
}
logZ_expanded_point = logZ__LME__(l, expanded_point, Beta);
if (logZ_expanded_point < logZ_reflected_point) {
for (int i = 0; i < Ndim; i++) {
simplex.nM[Nnodes_simplex - 1][i] = expanded_point.nV[i];
}
logZ.nV[Nnodes_simplex - 1] = logZ_expanded_point;
}
else {
for (int i = 0; i < Ndim; i++) {
simplex.nM[Nnodes_simplex - 1][i] = reflected_point.nV[i];
}
logZ.nV[Nnodes_simplex - 1] = logZ_reflected_point;
}
free__MatrixLib__(expanded_point);
}
static void contraction_Nelder_Mead__LME__(Matrix simplex, Matrix logZ,
Matrix reflected_point,
Matrix gravity_center, Matrix l,
double Beta,
double logZ_reflected_point) {
int Ndim = NumberDimensions;
int Nnodes_simplex = Ndim + 1;
double logZ_n1 = logZ.nV[Nnodes_simplex - 1];
double logZ_contracted_point;
Matrix contracted_point;
contracted_point = allocZ__MatrixLib__(1, Ndim);
if (logZ_reflected_point < logZ_n1) {
for (int i = 0; i < Ndim; i++) {
contracted_point.nV[i] =
gravity_center.nV[i] +
NM_gamma_LME * (reflected_point.nV[i] - gravity_center.nV[i]);
}
logZ_contracted_point = logZ__LME__(l, contracted_point, Beta);
if (logZ_contracted_point < logZ_reflected_point) {
for (int i = 0; i < Ndim; i++) {
simplex.nM[Nnodes_simplex - 1][i] = contracted_point.nV[i];
}
logZ.nV[Nnodes_simplex - 1] = logZ_contracted_point;
}
else {
shrinkage_Nelder_Mead__LME__(simplex, logZ, l, Beta);
}
}
else if (logZ_reflected_point > logZ_n1) {
for (int i = 0; i < Ndim; i++) {
contracted_point.nV[i] =
gravity_center.nV[i] -
NM_gamma_LME *
(gravity_center.nV[i] - simplex.nM[Nnodes_simplex - 1][i]);
}
logZ_contracted_point = logZ__LME__(l, contracted_point, Beta);
if (logZ_contracted_point < logZ_n1) {
for (int i = 0; i < Ndim; i++) {
simplex.nM[Nnodes_simplex - 1][i] = contracted_point.nV[i];
}
logZ.nV[Nnodes_simplex - 1] = logZ_contracted_point;
}
else {
shrinkage_Nelder_Mead__LME__(simplex, logZ, l, Beta);
}
}
free__MatrixLib__(contracted_point);
}
static void shrinkage_Nelder_Mead__LME__(Matrix simplex, Matrix logZ, Matrix l,
double Beta) {
int Ndim = NumberDimensions;
int Nnodes_simplex = Ndim + 1;
Matrix simplex_a = memory_to_matrix__MatrixLib__(1, Ndim, NULL);
for (int a = 0; a < Nnodes_simplex; a++) {
for (int i = 0; i < Ndim; i++) {
simplex.nM[a][i] = simplex.nM[0][i] +
NM_sigma_LME * (simplex.nM[a][i] - simplex.nM[0][i]);
}
simplex_a.nV = simplex.nM[a];
logZ.nV[a] = logZ__LME__(l, simplex_a, Beta);
}
}
static double fa__LME__(Matrix la,     
Matrix lambda, 
double Beta)   
{
int Ndim = NumberDimensions;
double la_x_la = 0.0;
double la_x_lambda = 0.0;
double fa = 0;
for (int i = 0; i < Ndim; i++) {
la_x_la += la.nV[i] * la.nV[i];
la_x_lambda += la.nV[i] * lambda.nV[i];
}
fa = -Beta * la_x_la + la_x_lambda;
return fa;
}
Matrix p__LME__(
Matrix l, 
Matrix lambda, 
double Beta)   
{
int N_a = l.N_rows;
int Ndim = NumberDimensions;
double Z = 0;
double Z_m1 = 0;
Matrix p = allocZ__MatrixLib__(1, N_a); 
Matrix la = memory_to_matrix__MatrixLib__(
1, Ndim, NULL); 
for (int a = 0; a < N_a; a++) {
la.nV = l.nM[a];
p.nV[a] = exp(fa__LME__(la, lambda, Beta));
Z += p.nV[a];
}
Z_m1 = (double)1 / Z;
for (int a = 0; a < N_a; a++) {
p.nV[a] *= Z_m1;
}
return p;
}
static double logZ__LME__(
Matrix l, 
Matrix lambda, 
double Beta)   
{
int N_a = l.N_rows;
int Ndim = NumberDimensions;
double Z = 0;
double logZ = 0;
Matrix la = memory_to_matrix__MatrixLib__(
1, Ndim, NULL); 
for (int a = 0; a < N_a; a++) {
la.nV = l.nM[a];
Z += exp(fa__LME__(la, lambda, Beta));
}
logZ = log(Z);
return logZ;
}
static Matrix r__LME__(
Matrix l, 
Matrix p) 
{
int N_a = l.N_rows;
int Ndim = NumberDimensions;
Matrix r = allocZ__MatrixLib__(Ndim, 1); 
for (int i = 0; i < Ndim; i++) {
for (int a = 0; a < N_a; a++) {
r.nV[i] += p.nV[a] * l.nM[a][i];
}
}
return r;
}
static Matrix J__LME__(
Matrix l, 
Matrix p, 
Matrix r) 
{
int N_a = l.N_rows;
int Ndim = NumberDimensions;
Matrix J = allocZ__MatrixLib__(Ndim, Ndim); 
for (int i = 0; i < Ndim; i++) {
for (int j = 0; j < Ndim; j++) {
for (int a = 0; a < N_a; a++) {
J.nM[i][j] += p.nV[a] * l.nM[a][i] * l.nM[a][j];
}
J.nM[i][j] -= r.nV[i] * r.nV[j];
}
}
return J;
}
Matrix dp__LME__(
Matrix l, 
Matrix p) 
{
int N_a = l.N_rows;
int Ndim = NumberDimensions;
Matrix dp = allocZ__MatrixLib__(N_a, Ndim);
Matrix r;      
Matrix J;      
Matrix Jm1;    
Matrix Jm1_la; 
Matrix la = memory_to_matrix__MatrixLib__(
Ndim, 1, NULL); 
r = r__LME__(l, p);
J = J__LME__(l, p, r);
Jm1 = inverse__MatrixLib__(J);
for (int a = 0; a < N_a; a++) {
la.nV = l.nM[a];
Jm1_la = matrix_product__MatrixLib__(Jm1, la);
for (int i = 0; i < Ndim; i++) {
dp.nM[a][i] = -p.nV[a] * Jm1_la.nV[i];
}
free__MatrixLib__(Jm1_la);
}
free__MatrixLib__(r);
free__MatrixLib__(J);
free__MatrixLib__(Jm1);
return dp;
}
int local_search__LME__(Particle MPM_Mesh, Mesh FEM_Mesh)
{
int STATUS = EXIT_SUCCESS;
int STATUS_p = EXIT_SUCCESS;  
int Ndim = NumberDimensions;
unsigned Np = MPM_Mesh.NumGP;
unsigned p;
Matrix X_p;           
Matrix dis_p;         
Matrix Delta_Xip;     
Matrix lambda_p;      
double Beta_p;        
ChainPtr Locality_I0; 
#pragma omp parallel shared(Np)
{
#pragma omp for private(p, X_p, dis_p, Locality_I0)
for (p = 0; p < Np; p++) {
X_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_GC.nM[p]);
dis_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.dis.nM[p]);
if (norm__MatrixLib__(dis_p, 2) > 0.0) {
Locality_I0 = FEM_Mesh.NodalLocality_0[MPM_Mesh.I0[p]];
MPM_Mesh.I0[p] = get_closest_node__MeshTools__(X_p, Locality_I0,
FEM_Mesh.Coordinates);
}
}
#pragma omp barrier
for (p = 0; p < Np; p++) {
int I0 = MPM_Mesh.I0[p];
Locality_I0 = FEM_Mesh.NodalLocality_0[MPM_Mesh.I0[p]];
while (Locality_I0 != NULL) {
if (FEM_Mesh.ActiveNode[Locality_I0->Idx] == false) {
FEM_Mesh.ActiveNode[Locality_I0->Idx] = true;
}
Locality_I0 = Locality_I0->next;
}
if ((Driver_EigenErosion == true) || (Driver_EigenSoftening == true)) {
push__SetLib__(&FEM_Mesh.List_Particles_Node[I0], p);
}
}
#pragma omp for private(p, X_p, Delta_Xip, lambda_p, Beta_p)
for (p = 0; p < Np; p++) {
lambda_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.lambda.nM[p]);
Beta_p = MPM_Mesh.Beta.nV[p]; 
X_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_GC.nM[p]);
free__SetLib__(&MPM_Mesh.ListNodes[p]);
MPM_Mesh.ListNodes[p] = NULL;
MPM_Mesh.ListNodes[p] =
tributary__LME__(p, X_p, Beta_p, MPM_Mesh.I0[p], FEM_Mesh);
MPM_Mesh.NumberNodes[p] = lenght__SetLib__(MPM_Mesh.ListNodes[p]);
Delta_Xip = compute_distance__MeshTools__(MPM_Mesh.ListNodes[p], X_p,
FEM_Mesh.Coordinates);
Beta_p = beta__LME__(gamma_LME, FEM_Mesh.h_avg[MPM_Mesh.I0[p]]);
MPM_Mesh.Beta.nV[p] = Beta_p;
STATUS_p = __lambda_Newton_Rapson(p, Delta_Xip, lambda_p, Beta_p);
if (STATUS_p == EXIT_FAILURE) {
fprintf(stderr, "" RED " Error in " RESET "" BOLDRED
"__lambda_Newton_Rapson() " RESET " \n");
STATUS = EXIT_FAILURE;
}
free__MatrixLib__(Delta_Xip);
}
}
if (STATUS == EXIT_FAILURE) {
return EXIT_FAILURE;
}
return EXIT_SUCCESS;
}
static ChainPtr tributary__LME__(int Indx_p, Matrix X_p, double Beta_p, int I0,
Mesh FEM_Mesh)
{
ChainPtr Triburary_Nodes = NULL;
int Ndim = NumberDimensions;
Matrix Distance; 
Matrix X_I = memory_to_matrix__MatrixLib__(Ndim, 1, NULL);
Matrix Metric = Identity__MatrixLib__(Ndim);
ChainPtr Set_Nodes0 = NULL;
int *Array_Nodes0;
int NumNodes0;
int Node0;
int NumTributaryNodes = 0;
double Ra = sqrt(-log(TOL_zero_LME) / Beta_p);
Set_Nodes0 = FEM_Mesh.NodalLocality[I0];
NumNodes0 = FEM_Mesh.SizeNodalLocality[I0];
Array_Nodes0 = set_to_memory__SetLib__(Set_Nodes0, NumNodes0);
for (int i = 0; i < NumNodes0; i++) {
Node0 = Array_Nodes0[i];
if (FEM_Mesh.ActiveNode[Node0] == true) {
X_I.nV = FEM_Mesh.Coordinates.nM[Node0];
Distance = substraction__MatrixLib__(X_p, X_I);
if (generalised_Euclidean_distance__MatrixLib__(Distance, Metric) <= Ra) {
push__SetLib__(&Triburary_Nodes, Node0);
NumTributaryNodes++;
}
free__MatrixLib__(Distance);
}
}
if (NumTributaryNodes < Ndim + 1) {
fprintf(stderr, "%s %i : %s -> %i\n",
"Warning in tributary__LME__ for particle", Indx_p,
"Insufficient nodal connectivity", NumTributaryNodes);
exit(EXIT_FAILURE);
}
free(Array_Nodes0);
free__MatrixLib__(Metric);
return Triburary_Nodes;
}
