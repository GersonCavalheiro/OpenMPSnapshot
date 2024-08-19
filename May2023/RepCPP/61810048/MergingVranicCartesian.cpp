
#include "MergingVranicCartesian.h"

#include <cmath>
#include <cstdlib>

MergingVranicCartesian::MergingVranicCartesian(Params& params,
Species * species, Random * rand)
: Merging(params, species, rand)
{
dimensions_[0] = (unsigned int)(species->merge_momentum_cell_size_[0]);
dimensions_[1] = (unsigned int)(species->merge_momentum_cell_size_[1]);
dimensions_[2] = (unsigned int)(species->merge_momentum_cell_size_[2]);

min_packet_size_ = species->merge_min_packet_size_;
max_packet_size_ = species->merge_max_packet_size_;

min_momentum_cell_length_[0] = species->merge_min_momentum_cell_length_[0];
min_momentum_cell_length_[1] = species->merge_min_momentum_cell_length_[1];
min_momentum_cell_length_[2] = species->merge_min_momentum_cell_length_[2];

accumulation_correction_ = species->merge_accumulation_correction_;

log_scale_ = species->merge_log_scale_;

min_momentum_log_scale_ = species->merge_min_momentum_log_scale_;

}

MergingVranicCartesian::~MergingVranicCartesian()
{
}


void MergingVranicCartesian::operator() (
double mass,
Particles &particles,
std::vector <int> &mask,
SmileiMPI* smpi,
int istart,
int iend,
int & count)
{

unsigned int number_of_particles = (unsigned int)(iend - istart);

if (number_of_particles > min_particles_per_cell_) {

unsigned int dim[3];
for (unsigned int i = 0; i < 3 ; i++) {
dim[i] = dimensions_[i];
}

double momentum_min[3];

double momentum_max[3];

double momentum_delta[3];

double inv_momentum_delta[3];

double cos_omega;
double sin_omega;

unsigned int mx_i;
unsigned int my_i;
unsigned int mz_i;

unsigned int ic;
unsigned int icc;
unsigned int ipack;
unsigned int npack;
unsigned int ip;
unsigned int ipr, ipr_min, ipr_max;

double total_weight;
double total_momentum_x;
double total_momentum_y;
double total_momentum_z;
double total_momentum_norm;
double total_energy;

double new_energy;
double new_momentum_norm;
double e1_x,e1_y,e1_z;
double e3_x,e3_y,e3_z;
double e2_x,e2_y,e2_z;
double e2_norm;

double cell_vec_x;
double cell_vec_y;
double cell_vec_z;


double * __restrict__ momentum_x = particles.getPtrMomentum(0);
double * __restrict__ momentum_y = particles.getPtrMomentum(1);
double * __restrict__ momentum_z = particles.getPtrMomentum(2);

double * __restrict__ weight = &( particles.weight( 0 ) );


unsigned int  * momentum_cell_index = new unsigned int [number_of_particles];

unsigned int  * sorted_particles = new unsigned int [number_of_particles];

double  * gamma = new double [number_of_particles];

if (mass == 0) {
#pragma omp simd private(ipr) 
for (ip=(unsigned int)(istart) ; ip<(unsigned int) (iend); ip++ ) {

ipr = ip - istart;

gamma[ipr] = sqrt(momentum_x[ip]*momentum_x[ip]
+ momentum_y[ip]*momentum_y[ip]
+ momentum_z[ip]*momentum_z[ip]);

}
} else {
#pragma omp simd private(ipr) 
for (ip=(unsigned int)(istart) ; ip<(unsigned int) (iend); ip++ ) {

ipr = ip - istart;

gamma[ipr] = sqrt(1.0 + momentum_x[ip]*momentum_x[ip]
+ momentum_y[ip]*momentum_y[ip]
+ momentum_z[ip]*momentum_z[ip]);

}
}

momentum_min[0] = momentum_x[istart];
momentum_max[0] = momentum_x[istart];

momentum_min[1] = momentum_y[istart];
momentum_max[1] = momentum_y[istart];

momentum_min[2] = momentum_z[istart];
momentum_max[2] = momentum_z[istart];

#if __INTEL_COMPILER > 18000
#pragma omp simd \
reduction(min:momentum_min)  \
reduction(max:momentum_max)
#endif
for (ip=(unsigned int) (istart) ; ip < (unsigned int) (iend); ip++ ) {
momentum_min[0] = std::min(momentum_min[0],momentum_x[ip]);
momentum_max[0] = std::max(momentum_max[0],momentum_x[ip]);

momentum_min[1] = std::min(momentum_min[1],momentum_y[ip]);
momentum_max[1] = std::max(momentum_max[1],momentum_y[ip]);

momentum_min[2] = std::min(momentum_min[2],momentum_z[ip]);
momentum_max[2] = std::max(momentum_max[2],momentum_z[ip]);
}



for (ip = 0 ; ip < 3 ; ip++) {
if (fabs((momentum_max[ip] - momentum_min[ip])) < min_momentum_cell_length_[ip]) {
if (momentum_max[ip] <= 0 || momentum_min[ip] >= 0) {
momentum_delta[ip] = min_momentum_cell_length_[ip];
momentum_min[ip] = (momentum_max[ip] + momentum_min[ip] - momentum_delta[ip])*0.5;
momentum_max[ip] = (momentum_max[ip] + momentum_min[ip] + momentum_delta[ip])*0.5;
inv_momentum_delta[ip] = 0;
dim[ip] = 1;
} else {
momentum_max[ip] = std::max(fabs((momentum_max[ip] + momentum_min[ip] + min_momentum_cell_length_[ip])*0.5),fabs((momentum_max[ip] + momentum_min[ip] - min_momentum_cell_length_[ip])*0.5));
momentum_min[ip] = -momentum_max[ip];
momentum_delta[ip] = momentum_max[ip];
inv_momentum_delta[ip] = 1.0/momentum_delta[ip];
dim[ip] = 2;
}
} else {
if (dim[ip] == 1) {
momentum_max[ip] += (momentum_max[ip] - momentum_min[ip])*momentum_max_factor_;
momentum_delta[ip] = (momentum_max[ip] - momentum_min[ip]);
inv_momentum_delta[ip] = 1.0/momentum_delta[ip];
} else if (momentum_max[ip] <= 0 || momentum_min[ip] >= 0) {
momentum_max[ip] += (momentum_max[ip] - momentum_min[ip])*momentum_max_factor_;
if (accumulation_correction_) {
momentum_delta[ip] = (momentum_max[ip] - momentum_min[ip]) / (dim[ip]-1);
momentum_min[ip] -= 0.99*momentum_delta[ip]*rand_->uniform();
} else {
momentum_delta[ip] = (momentum_max[ip] - momentum_min[ip]) / (dim[ip]);
}
inv_momentum_delta[ip] = 1.0/momentum_delta[ip];
} else {
if (accumulation_correction_) {
dim[ip] = int(dim[ip]*(1+rand_->uniform()));
}


momentum_max[ip] += (momentum_max[ip] - momentum_min[ip])*momentum_max_factor_;

momentum_delta[ip] = fabs(momentum_max[ip] - momentum_min[ip]) / dim[ip];
inv_momentum_delta[ip] = 1.0/momentum_delta[ip];
double nb_delta_min = (ceil(fabs(momentum_min[ip]) * inv_momentum_delta[ip]));
double nb_delta_max = (ceil(fabs(momentum_max[ip]) * inv_momentum_delta[ip]));
momentum_min[ip] = - nb_delta_min * momentum_delta[ip];
momentum_max[ip] =   nb_delta_max * momentum_delta[ip];
dim[ip] = (unsigned int)(nb_delta_max + nb_delta_min);


}
}
}






unsigned int momentum_cells = dim[0]
* dim[1]
* dim[2];

unsigned int  * particles_per_momentum_cells = new unsigned int [momentum_cells];

unsigned int  * momentum_cell_particle_index = new unsigned int [momentum_cells];

#pragma omp simd
for (ic = 0 ; ic < momentum_cells ; ic++) {
momentum_cell_particle_index[ic] = 0;
particles_per_momentum_cells[ic] = 0;
}

#pragma omp simd \
private(ipr,mx_i,my_i,mz_i) 
for (ip=(unsigned int) (istart) ; ip < (unsigned int) (iend); ip++ ) {

ipr = ip - istart;

mx_i = (unsigned int) floor( (momentum_x[ip] - momentum_min[0]) * inv_momentum_delta[0]);
my_i = (unsigned int) floor( (momentum_y[ip] - momentum_min[1]) * inv_momentum_delta[1]);
mz_i = (unsigned int) floor( (momentum_z[ip] - momentum_min[2]) * inv_momentum_delta[2]);

momentum_cell_index[ipr] = mz_i * dim[0]*dim[1]
+ my_i * dim[0] + mx_i;


}


for (ipr=0; ipr<number_of_particles; ipr++ ) {
particles_per_momentum_cells[momentum_cell_index[ipr]] += 1;
}


for (ic = 1 ; ic < momentum_cells ; ic++) {
momentum_cell_particle_index[ic]  = momentum_cell_particle_index[ic-1] + particles_per_momentum_cells[ic-1];
particles_per_momentum_cells[ic-1] = 0;
}
particles_per_momentum_cells[momentum_cells-1] = 0;


for (ipr=0 ; ipr<number_of_particles; ipr++ ) {

ic = momentum_cell_index[ipr];


sorted_particles[momentum_cell_particle_index[ic]
+ particles_per_momentum_cells[ic]] = istart + ipr;

particles_per_momentum_cells[ic] += 1;
}



for (mz_i=0 ; mz_i< dim[2]; mz_i++ ) {

cell_vec_z = momentum_min[2] + (mz_i+0.5)*momentum_delta[2];

for (my_i=0 ; my_i< dim[1]; my_i++ ) {

cell_vec_y = momentum_min[1] + (my_i+0.5)*momentum_delta[1];

icc = my_i + mz_i* dim[1] ;

for (mx_i=0 ; mx_i< dim[0]; mx_i++ ) {

cell_vec_x = momentum_min[0] + (mx_i+0.5)*momentum_delta[0];

ic = mx_i + icc*dim[0];

if (particles_per_momentum_cells[ic] >= min_packet_size_ ) {

npack = particles_per_momentum_cells[ic]/max_packet_size_;

if (particles_per_momentum_cells[ic]%max_packet_size_ >= min_packet_size_) {
npack += 1;
}

for (ipack = 0 ; ipack < npack ; ipack += 1) {

total_weight = 0;
total_momentum_x = 0;
total_momentum_y = 0;
total_momentum_z = 0;
total_energy = 0;

ipr_min = ipack*max_packet_size_;
ipr_max = std::min((ipack+1)*max_packet_size_,particles_per_momentum_cells[ic]);



for (ipr = ipr_min ; ipr < ipr_max ; ipr ++) {

ip = sorted_particles[momentum_cell_particle_index[ic] + ipr];

total_weight += weight[ip];

total_momentum_x += momentum_x[ip]*weight[ip];
total_momentum_y += momentum_y[ip]*weight[ip];
total_momentum_z += momentum_z[ip]*weight[ip];

total_energy += weight[ip]*gamma[ip - istart];

}

new_energy = total_energy / total_weight;

if (mass == 0) {
new_momentum_norm = new_energy;
} else {
new_momentum_norm = sqrt(new_energy*new_energy - 1.0);
}

total_momentum_norm = sqrt(total_momentum_x*total_momentum_x
+      total_momentum_y*total_momentum_y
+      total_momentum_z*total_momentum_z);

cos_omega = std::min(total_momentum_norm / (total_weight*new_momentum_norm),1.0);
sin_omega = sqrt(1 - cos_omega*cos_omega);

total_momentum_norm = 1/total_momentum_norm;

e1_x = total_momentum_x*total_momentum_norm;
e1_y = total_momentum_y*total_momentum_norm;
e1_z = total_momentum_z*total_momentum_norm;

e3_x = e1_y*cell_vec_z - e1_z*cell_vec_y;
e3_y = e1_z*cell_vec_x - e1_x*cell_vec_z;
e3_z = e1_x*cell_vec_y - e1_y*cell_vec_x;

if (fabs(e3_x*e3_x + e3_y*e3_y + e3_z*e3_z) > 0)
{


e2_x = e1_y*e3_z - e1_z*e3_y;
e2_y = e1_z*e3_x - e1_x*e3_z;
e2_z = e1_x*e3_y - e1_y*e3_x;

e2_norm = 1./sqrt(e2_x*e2_x + e2_y*e2_y + e2_z*e2_z);

e2_x = e2_x * e2_norm;
e2_y = e2_y * e2_norm;
e2_z = e2_z * e2_norm;




ip = sorted_particles[momentum_cell_particle_index[ic] + ipr_min];

momentum_x[ip] = new_momentum_norm*(cos_omega*e1_x + sin_omega*e2_x);
momentum_y[ip] = new_momentum_norm*(cos_omega*e1_y + sin_omega*e2_y);
momentum_z[ip] = new_momentum_norm*(cos_omega*e1_z + sin_omega*e2_z);
weight[ip] = 0.5*total_weight;

ip = sorted_particles[momentum_cell_particle_index[ic] + ipr_min + 1];
momentum_x[ip] = new_momentum_norm*(cos_omega*e1_x - sin_omega*e2_x);
momentum_y[ip] = new_momentum_norm*(cos_omega*e1_y - sin_omega*e2_y);
momentum_z[ip] = new_momentum_norm*(cos_omega*e1_z - sin_omega*e2_z);
weight[ip] = 0.5*total_weight;

for (ipr = ipr_min + 2; ipr < ipr_max ; ipr ++) {
ip = sorted_particles[momentum_cell_particle_index[ic] + ipr];
mask[ip] = -1;
count--;
}



} else {

if (mass == 0) {

ip = sorted_particles[momentum_cell_particle_index[ic] + ipr_min];

momentum_x[ip] = new_momentum_norm*e1_x;
momentum_y[ip] = new_momentum_norm*e1_y;
momentum_z[ip] = new_momentum_norm*e1_z;
weight[ip] = total_weight;

for (ipr = ipr_min + 1; ipr < ipr_max ; ipr ++) {
ip = sorted_particles[momentum_cell_particle_index[ic] + ipr];
mask[ip] = -1;
count--;
}


}

}

}
}

}
}
}


delete [] gamma;
delete [] momentum_cell_index;
delete [] sorted_particles;
delete [] particles_per_momentum_cells;
delete [] momentum_cell_particle_index;
}

}
