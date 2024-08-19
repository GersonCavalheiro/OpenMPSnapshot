#include "Nodes/Q4.h"
extern double Thickness_Plain_Stress;
static Matrix F_Ref__Q4__(Matrix, Matrix);
static Matrix Xi_to_X__Q4__(Matrix, Matrix);
static void X_to_Xi__Q4__(Matrix, Matrix, Matrix);
void initialize__Q4__(Particle MPM_Mesh, Mesh FEM_Mesh) {
int Ndim = NumberDimensions;
int Np = MPM_Mesh.NumGP;
int Nelem = FEM_Mesh.NumElemMesh;
bool Is_particle_reachable;
unsigned p;
#pragma omp parallel shared(Np, Nelem)
{
#pragma omp for private(p, Is_particle_reachable)
for (p = 0; p < Np; p++) {
Is_particle_reachable = false;
unsigned idx_element = 0;
MPM_Mesh.NumberNodes[p] = 4;
Matrix X_p =
memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_GC.nM[p]);
Matrix Xi_p =
memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_EC.nM[p]);
while ((Is_particle_reachable == false) && (idx_element < Nelem)) {
ChainPtr Elem_p_Connectivity = FEM_Mesh.Connectivity[idx_element];
Matrix Elem_p_Coordinates = get_nodes_coordinates__MeshTools__(
Elem_p_Connectivity, FEM_Mesh.Coordinates);
if (in_out__Q4__(X_p, Elem_p_Coordinates) == true) {
Is_particle_reachable = true;
MPM_Mesh.Element_p[p] = idx_element;
MPM_Mesh.I0[p] = get_closest_node__MeshTools__(
X_p, Elem_p_Connectivity, FEM_Mesh.Coordinates);
MPM_Mesh.ListNodes[p] = copy__SetLib__(Elem_p_Connectivity);
X_to_Xi__Q4__(Xi_p, X_p, Elem_p_Coordinates);
}
free__MatrixLib__(Elem_p_Coordinates);
++idx_element;
}
if (!Is_particle_reachable) {
fprintf(stderr, "%s : %s %i\n", "Error in initialize__Q4__()",
"The search algorithm was unable to find particle", p);
exit(EXIT_FAILURE);
}
}
}
ChainPtr Locality_I0; 
int I0;
for (unsigned p = 0; p < Np; p++) {
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
}
Matrix N__Q4__(Matrix X_e) {
Matrix N_ref = allocZ__MatrixLib__(1, 4);
N_ref.nV[0] = 0.25 * (1 - X_e.nV[0]) * (1 - X_e.nV[1]);
N_ref.nV[1] = 0.25 * (1 + X_e.nV[0]) * (1 - X_e.nV[1]);
N_ref.nV[2] = 0.25 * (1 + X_e.nV[0]) * (1 + X_e.nV[1]);
N_ref.nV[3] = 0.25 * (1 - X_e.nV[0]) * (1 + X_e.nV[1]);
return N_ref;
}
Matrix dN_Ref__Q4__(Matrix X_e) {
int Ndim = NumberDimensions;
Matrix dNdX_ref = allocZ__MatrixLib__(4, Ndim);
dNdX_ref.nM[0][0] = -0.25 * (1 - X_e.nV[1]);
dNdX_ref.nM[0][1] = -0.25 * (1 - X_e.nV[0]);
dNdX_ref.nM[1][0] = +0.25 * (1 - X_e.nV[1]);
dNdX_ref.nM[1][1] = -0.25 * (1 + X_e.nV[0]);
dNdX_ref.nM[2][0] = +0.25 * (1 + X_e.nV[1]);
dNdX_ref.nM[2][1] = +0.25 * (1 + X_e.nV[0]);
dNdX_ref.nM[3][0] = -0.25 * (1 + X_e.nV[1]);
dNdX_ref.nM[3][1] = +0.25 * (1 - X_e.nV[0]);
return dNdX_ref;
}
static Matrix F_Ref__Q4__(Matrix X_NC_GP, Matrix X_GC_Nodes)
{
int Ndim = NumberDimensions;
Matrix dNdX_Ref_GP;
Tensor X_I;
Tensor dNdx_I;
Tensor F_Ref_I;
Matrix F_Ref = allocZ__MatrixLib__(Ndim, Ndim);
dNdX_Ref_GP = dN_Ref__Q4__(X_NC_GP);
for (int I = 0; I < 4; I++) {
X_I = memory_to_tensor__TensorLib__(X_GC_Nodes.nM[I], 1);
dNdx_I = memory_to_tensor__TensorLib__(dNdX_Ref_GP.nM[I], 1);
F_Ref_I = dyadic_Product__TensorLib__(X_I, dNdx_I);
for (int i = 0; i < Ndim; i++) {
for (int j = 0; j < Ndim; j++) {
F_Ref.nM[i][j] += F_Ref_I.N[i][j];
}
}
free__TensorLib__(F_Ref_I);
}
free__MatrixLib__(dNdX_Ref_GP);
return F_Ref;
}
Matrix dN__Q4__(Matrix Xi, Matrix Element)
{
Matrix dNdX;
Matrix dNdX_T;
Matrix dNdX_Ref = dN_Ref__Q4__(Xi);
Matrix dNdX_Ref_T = transpose__MatrixLib__(dNdX_Ref);
Matrix F = F_Ref__Q4__(Xi, Element);
Matrix F_m1 = inverse__MatrixLib__(F);
Matrix F_Tm1 = transpose__MatrixLib__(F_m1);
dNdX_T = matrix_product__MatrixLib__(F_Tm1, dNdX_Ref_T);
dNdX = transpose__MatrixLib__(dNdX_T);
free__MatrixLib__(F);
free__MatrixLib__(F_m1);
free__MatrixLib__(F_Tm1);
free__MatrixLib__(dNdX_Ref);
free__MatrixLib__(dNdX_Ref_T);
free__MatrixLib__(dNdX_T);
return dNdX;
}
Matrix Xi_to_X__Q4__(Matrix Xi, Matrix Element)
{
int Ndim = NumberDimensions;
Matrix N = N__Q4__(Xi);
Matrix X = allocZ__MatrixLib__(Ndim, 1);
for (int I = 0; I < 4; I++) {
for (int i = 0; i < Ndim; i++) {
X.nV[i] += N.nV[I] * Element.nM[I][i];
}
}
free__MatrixLib__(N);
return X;
}
static void X_to_Xi__Q4__(Matrix Xi, Matrix X, Matrix Element)
{
Xi = Newton_Rapson(Xi_to_X__Q4__, Element, F_Ref__Q4__, Element, X, Xi);
}
bool in_out__Q4__(Matrix X, Matrix Element) {
double min[2] = {Element.nM[0][0], Element.nM[0][1]};
double max[2] = {Element.nM[0][0], Element.nM[0][1]};
for (int a = 1; a < 4; a++) {
for (int i = 0; i < 2; i++) {
min[i] = DMIN(min[i], Element.nM[a][i]);
max[i] = DMAX(max[i], Element.nM[a][i]);
}
}
Matrix Xi;
if ((X.nV[0] <= max[0]) && (X.nV[0] >= min[0]) && (X.nV[1] <= max[1]) &&
(X.nV[1] >= min[1])) {
Xi = allocZ__MatrixLib__(2, 1);
X_to_Xi__Q4__(Xi, X, Element);
if ((fabs(Xi.nV[0]) <= 1.0) && (fabs(Xi.nV[1]) <= 1.0)) {
free__MatrixLib__(Xi);
return true;
} else {
free__MatrixLib__(Xi);
return false;
}
} else {
return false;
}
}
void element_to_particles__Q4__(Matrix X_p, Mesh FEM_Mesh, int GPxElement) {
int Ndim = NumberDimensions;
int NumElemMesh = FEM_Mesh.NumElemMesh;
int NumNodesElem = 4;
Matrix N_GP;
Matrix Xi_p = allocZ__MatrixLib__(GPxElement, Ndim);
Matrix Xi_p_j = memory_to_matrix__MatrixLib__(1, Ndim, NULL);
Element Element;
int Node;
switch (GPxElement) {
case 1:
Xi_p.nV[0] = 0.0;
Xi_p.nV[1] = 0.0;
break;
case 4:
Xi_p.nM[0][0] = 1. / sqrt(3.0);
Xi_p.nM[0][1] = 1. / sqrt(3.0);
Xi_p.nM[1][0] = 1. / sqrt(3.0);
Xi_p.nM[1][1] = -1. / sqrt(3.0);
Xi_p.nM[2][0] = -1. / sqrt(3.0);
Xi_p.nM[2][1] = 1. / sqrt(3.0);
Xi_p.nM[3][0] = -1. / sqrt(3.0);
Xi_p.nM[3][1] = -1. / sqrt(3.0);
break;
case 5:
Xi_p.nM[0][0] = 0.5;
Xi_p.nM[0][1] = 0.5;
Xi_p.nM[1][0] = 0.5;
Xi_p.nM[1][1] = -0.5;
Xi_p.nM[2][0] = -0.5;
Xi_p.nM[2][1] = 0.5;
Xi_p.nM[3][0] = -0.5;
Xi_p.nM[3][1] = -0.5;
Xi_p.nM[4][0] = 0.0;
Xi_p.nM[4][1] = 0.0;
break;
case 9:
Xi_p.nM[0][0] = 0.0;
Xi_p.nM[0][1] = 0.0;
Xi_p.nM[1][0] = 0.6666666666666;
Xi_p.nM[1][1] = 0.0;
Xi_p.nM[2][0] = 0.6666666666666;
Xi_p.nM[2][1] = 0.6666666666666;
Xi_p.nM[3][0] = 0.0;
Xi_p.nM[3][1] = 0.6666666666666;
Xi_p.nM[4][0] = -0.6666666666666;
Xi_p.nM[4][1] = 0.6666666666666;
Xi_p.nM[5][0] = -0.6666666666666;
Xi_p.nM[5][1] = 0.0;
Xi_p.nM[6][0] = -0.6666666666666;
Xi_p.nM[6][1] = -0.6666666666666;
Xi_p.nM[7][0] = 0.0;
Xi_p.nM[7][1] = -0.6666666666666;
Xi_p.nM[8][0] = 0.6666666666666;
Xi_p.nM[8][1] = -0.6666666666666;
break;
default:
fprintf(stderr, "%s : %s \n", "Error in element_to_particles__Q4__()",
"Wrong number of particles per element");
exit(EXIT_FAILURE);
}
for (int i = 0; i < NumElemMesh; i++) {
Element = nodal_set__Particles__(i, FEM_Mesh.Connectivity[i],
FEM_Mesh.NumNodesElem[i]);
for (int j = 0; j < GPxElement; j++) {
if (GPxElement == 1) {
N_GP = N__Q4__(Xi_p);
} else {
Xi_p_j.nV = Xi_p.nM[j];
N_GP = N__Q4__(Xi_p_j);
}
for (int k = 0; k < NumNodesElem; k++) {
Node = Element.Connectivity[k];
for (int l = 0; l < Ndim; l++) {
X_p.nM[i * GPxElement + j][l] +=
N_GP.nV[k] * FEM_Mesh.Coordinates.nM[Node][l];
}
}
free__MatrixLib__(N_GP);
}
free(Element.Connectivity);
}
free__MatrixLib__(Xi_p);
}
double min_DeltaX__Q4__(ChainPtr Element_Connectivity, Matrix Coordinates) {
int Ndim = NumberDimensions;
int Node_k;
int Node_k1;
int NumNodesElem = 4;
int NumSides = 4;
double sqr_lenght = 0.0;
double MinElementSize = 10e16;
for (int k = 0; k < NumNodesElem; k++) {
Node_k = Element_Connectivity->Idx;
Node_k1 = Element_Connectivity->next->Idx;
sqr_lenght = 0.0;
for (int l = 0; l < Ndim; l++) {
sqr_lenght +=
DSQR(Coordinates.nM[Node_k1][l] - Coordinates.nM[Node_k][l]);
}
MinElementSize = DMIN(MinElementSize, sqrt(sqr_lenght));
Element_Connectivity = Element_Connectivity->next;
}
return MinElementSize;
}
double volume__Q4__(Matrix Element) {
int Ndim = NumberDimensions;
double J_i;
double Vol = 0;
double table_w[4] = {1., 1., 1., 1.};
double table_X[4][2] = {{-0.577350269200000, -0.577350269200000},
{0.577350269200000, -0.577350269200000},
{-0.577350269200000, 0.577350269200000},
{0.577350269200000, 0.577350269200000}};
Matrix F_i;
Matrix Xi = allocZ__MatrixLib__(2, 1);
for (int i = 0; i < 4; i++) {
for (int j = 0; j < Ndim; j++) {
Xi.nV[j] = table_X[i][j];
}
F_i = F_Ref__Q4__(Xi, Element);
J_i = I3__MatrixLib__(F_i);
Vol += fabs(J_i) * table_w[i];
free__MatrixLib__(F_i);
}
Vol *= Thickness_Plain_Stress;
free__MatrixLib__(Xi);
return Vol;
}
void local_search__Q4__(Particle MPM_Mesh, Mesh FEM_Mesh) {
int Ndim = NumberDimensions;
unsigned Np = MPM_Mesh.NumGP;
unsigned p;
Matrix Xi_p;
Matrix X_p;
Matrix dis_p;
int I0_p_old;
int I0_p_new;
ChainPtr Connectivity_p;
Matrix CoordElement;
ChainPtr Locality_I0;
#pragma omp parallel shared(Np)
{
#pragma omp for private(p, Xi_p, X_p, dis_p, I0_p_old, I0_p_new,               Connectivity_p, CoordElement, Locality_I0)
for (p = 0; p < Np; p++) {
X_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_GC.nM[p]);
dis_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.dis.nM[p]);
Xi_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_EC.nM[p]);
if (norm__MatrixLib__(dis_p, 2) > 0) {
I0_p_old = MPM_Mesh.I0[p];
Locality_I0 = FEM_Mesh.NodalLocality_0[I0_p_old];
I0_p_new = get_closest_node__MeshTools__(X_p, Locality_I0,
FEM_Mesh.Coordinates);
MPM_Mesh.I0[p] = I0_p_new;
MPM_Mesh.Element_p[p] =
search_particle_in_surrounding_elements__Particles__(
p, X_p, FEM_Mesh.NodeNeighbour[I0_p_new], FEM_Mesh);
if (MPM_Mesh.Element_p[p] == -999) {
fprintf(stderr, "%s : %s %i \n",
"Error in local_search__Q4__ -> "
"search_particle_in_surrounding_elements__Particles__",
"Not posible to find the particle", p);
}
free__SetLib__(&MPM_Mesh.ListNodes[p]);
MPM_Mesh.ListNodes[p] = NULL;
MPM_Mesh.ListNodes[p] =
copy__SetLib__(FEM_Mesh.Connectivity[MPM_Mesh.Element_p[p]]);
CoordElement = get_nodes_coordinates__MeshTools__(MPM_Mesh.ListNodes[p],
FEM_Mesh.Coordinates);
X_to_Xi__Q4__(Xi_p, X_p, CoordElement);
free__MatrixLib__(CoordElement);
}
}
}
for (p = 0; p < Np; p++) {
int I0 = MPM_Mesh.I0[p];
Connectivity_p = MPM_Mesh.ListNodes[p];
while (Connectivity_p != NULL) {
if (FEM_Mesh.ActiveNode[Connectivity_p->Idx] == false) {
FEM_Mesh.ActiveNode[Connectivity_p->Idx] = true;
}
Connectivity_p = Connectivity_p->next;
}
if ((Driver_EigenErosion == true) || (Driver_EigenSoftening == true)) {
push__SetLib__(&FEM_Mesh.List_Particles_Node[I0], p);
}
}
}
