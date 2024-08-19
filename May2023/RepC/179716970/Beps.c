#include "Constitutive/Fracture/Beps.h"
#include "Globals.h"
void compute_Beps__Constitutive__(Particle MPM_Mesh, Mesh FEM_Mesh,
bool Initialize_Beps) {
unsigned Np = MPM_Mesh.NumGP;
unsigned p;
double DeltaX = FEM_Mesh.DeltaX;
#pragma omp parallel
{
#pragma omp for private(p)
for (p = 0; p < Np; p++) {
const double *dis_p = MPM_Mesh.Phi.dis.nM[p];
if ((euclidean_norm__TensorLib__(dis_p) > 0.000001) ||
(Initialize_Beps == true)) {
free__SetLib__(&MPM_Mesh.Beps[p]);
} else {
continue;
}
unsigned MatIndx_p = MPM_Mesh.MatIdx[p];
Material MatProp_p = MPM_Mesh.Mat[MatIndx_p];
double Ceps_p = MatProp_p.Ceps;
double eps_distance_p = Ceps_p * DeltaX;
const double *xp = MPM_Mesh.Phi.x_GC.nM[p];
unsigned I0_p = MPM_Mesh.I0[p];
ChainPtr Nodal_Locality_p = FEM_Mesh.NodalLocality_0[I0_p];
while (Nodal_Locality_p != NULL) {
unsigned A = Nodal_Locality_p->Idx;
ChainPtr Particle_Locality_I = FEM_Mesh.List_Particles_Node[A];
while (Particle_Locality_I != NULL) {
unsigned q = Particle_Locality_I->Idx;
const double *xq = MPM_Mesh.Phi.x_GC.nM[q];
if (euclidean_distance__TensorLib__(xp, xq) <= eps_distance_p) {
push__SetLib__(&MPM_Mesh.Beps[p], q);
}
Particle_Locality_I = Particle_Locality_I->next;
} 
Nodal_Locality_p = Nodal_Locality_p->next;
} 
} 
}   
}
