
#include "MergingVranicSpherical.h"
#include <cmath>

MergingVranicSpherical::MergingVranicSpherical(Params& params,
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

MergingVranicSpherical::~MergingVranicSpherical()
{
}


void MergingVranicSpherical::operator() (
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


unsigned int mr_dim = dimensions_[0];
unsigned int theta_dim_ref = dimensions_[1];
unsigned int theta_dim_min = 1;
unsigned int phi_dim = dimensions_[2];
unsigned int  * theta_dim = new unsigned int [phi_dim];

double mr_min;
double theta_min_ref;
double  * theta_min = new double [phi_dim];
double phi_min;

double mr_max;
double theta_max_ref;
double  * theta_max = new double[phi_dim];
double phi_max;

double mr_delta;
double theta_delta_ref;
double  * theta_delta = new double[phi_dim];
double phi_delta;

double inv_mr_delta;
double  * inv_theta_delta = new double [phi_dim];
double inv_phi_delta;

double mr_interval;
double theta_interval;
double phi_interval;

double phi;
double theta;
double cos_omega;
double sin_omega;

unsigned int mr_i;
unsigned int theta_i;
unsigned int phi_i;

unsigned int ic, icc;
unsigned int ipack;
unsigned int npack;
unsigned int ipr, ipr_min, ipr_max;
unsigned int ip;

double total_weight;
double total_energy;

double new_energy;
double new_momentum_norm;
double e1_x,e1_y,e1_z;
double e2_x,e2_y,e2_z;
double e3_x,e3_y,e3_z;
double e2_norm;

double* momentum[3];
for ( int i = 0 ; i<3 ; i++ )
momentum[i] =  &( particles.momentum(i,0) );

double * __restrict__ momentum_x = particles.getPtrMomentum(0);
double * __restrict__ momentum_y = particles.getPtrMomentum(1);
double * __restrict__ momentum_z = particles.getPtrMomentum(2);

double *weight = &( particles.weight( 0 ) );


double  * momentum_norm = new double [number_of_particles];

unsigned int  * momentum_cell_index = new unsigned int [number_of_particles];

unsigned int  * sorted_particles = new unsigned int [number_of_particles];

double  * particles_phi = new double [number_of_particles];
double  * particles_theta = new double [number_of_particles];

#pragma omp simd private(ipr)
for (ip=(unsigned int)(istart) ; ip<(unsigned int) (iend); ip++ ) {

ipr = ip - istart;

momentum_norm[ipr] = sqrt(momentum_x[ip]*momentum_x[ip]
+ momentum_y[ip]*momentum_y[ip]
+ momentum_z[ip]*momentum_z[ip]);

particles_phi[ipr]   = asin(momentum_z[ip] / momentum_norm[ipr]);
particles_theta[ipr] = atan2(momentum_y[ip] , momentum_x[ip]);
}

mr_min = momentum_norm[0];
mr_max = mr_min;

theta_min_ref = particles_theta[0];
phi_min   = particles_phi[0];

theta_max_ref = theta_min_ref;
phi_max   = phi_min;

#pragma omp simd \
reduction(min:mr_min) reduction(min:theta_min_ref) reduction(min:phi_min) \
reduction(max:mr_max) reduction(max:theta_max_ref) reduction(max:phi_max)
for (ipr=1 ; ipr < number_of_particles; ipr++ ) {
mr_min = std::min(mr_min,momentum_norm[ipr]);
mr_max = std::max(mr_max,momentum_norm[ipr]);

theta_min_ref = std::min(theta_min_ref,particles_theta[ipr]);
theta_max_ref = std::max(theta_max_ref,particles_theta[ipr]);

phi_min = std::min(phi_min,particles_phi[ipr]);
phi_max = std::max(phi_max,particles_phi[ipr]);
}

if (log_scale_) {
mr_min = std::max(mr_min,min_momentum_log_scale_);
mr_max = std::max(mr_max,min_momentum_log_scale_);
mr_interval = std::abs(mr_max - mr_min);
if (mr_interval < min_momentum_cell_length_[0]) {
mr_delta = min_momentum_cell_length_[0];
inv_mr_delta = 0;
mr_dim = 1;
mr_min = log10(mr_min);
mr_max = log10(mr_max);
} else {
mr_max += (mr_interval)*0.01;
mr_min = log10(mr_min);
mr_max = log10(mr_max);
mr_delta = std::abs(mr_max - mr_min) / (mr_dim);
inv_mr_delta = 1./mr_delta;
}
} else {
mr_interval = std::abs(mr_max - mr_min);
if (mr_interval < min_momentum_cell_length_[0]) {
mr_delta = min_momentum_cell_length_[0];
inv_mr_delta = 0;
mr_dim = 1;
} else {
if (mr_dim == 1) {
mr_max += (mr_interval)*0.01;
mr_interval = std::abs(mr_max - mr_min);
mr_delta = mr_interval;
inv_mr_delta = 1./mr_delta;
} else {
if (accumulation_correction_) {
mr_delta = (mr_interval) / (mr_dim-1);
mr_min -= 0.99*mr_delta*rand_->uniform();
inv_mr_delta = 1./mr_delta;
} else {
mr_max += (mr_interval)*0.01;
mr_delta = (mr_max - mr_min) / (mr_dim);
inv_mr_delta = 1./mr_delta;
}
}
}
}

phi_interval = std::abs(phi_max - phi_min);
if (phi_interval < min_momentum_cell_length_[2]) {
phi_delta = min_momentum_cell_length_[2];
inv_phi_delta = 0;
phi_dim = 1;
} else {
if (phi_dim == 1) {
phi_max += (phi_max - phi_min)*0.01;
phi_delta = (phi_max - phi_min);
inv_phi_delta = 1./phi_delta;
}
else {
if (accumulation_correction_) {
phi_delta = (phi_interval) / (phi_dim-1);
phi_min -= 0.99*phi_delta*rand_->uniform();
inv_phi_delta = 1./phi_delta;
} else {
phi_max += (phi_interval)*0.01;
phi_delta = (phi_max - phi_min) / (phi_dim);
inv_phi_delta = 1./phi_delta;
}
}
}

if (std::abs(theta_max_ref - theta_min_ref) < min_momentum_cell_length_[1]) {
theta_delta_ref = min_momentum_cell_length_[1];
theta_interval  = min_momentum_cell_length_[1];
theta_dim_ref = 1;
} else {
if (theta_dim_ref == 1) {
theta_max_ref  += (theta_max_ref - theta_min_ref)*0.01;
theta_delta_ref = std::abs(theta_max_ref - theta_min_ref);
theta_interval  = std::abs(theta_max_ref - theta_min_ref);
}
else {
if (accumulation_correction_) {
theta_delta_ref = std::abs(theta_max_ref - theta_min_ref) / (theta_dim_ref);
theta_interval = std::abs(theta_max_ref - theta_min_ref);
theta_dim_min  = std::max((int)(ceil(2 * theta_interval / M_PI)),2);
} else {
theta_max_ref  += (theta_max_ref - theta_min_ref)*0.01;
theta_interval = std::abs(theta_max_ref - theta_min_ref);
theta_delta_ref = theta_interval / (theta_dim_ref);
theta_dim_min  = std::max((int)(ceil(2 * theta_interval / M_PI)),2);
}
}
}
double absolute_phi_min = 0.5*M_PI;
for(phi_i=0 ; phi_i < phi_dim ; phi_i++) {
absolute_phi_min = std::min(std::abs((phi_i + 0.5)*phi_delta + phi_min),absolute_phi_min);
}
for(phi_i=0 ; phi_i < phi_dim ; phi_i++) {
if (theta_dim_ref == 1) {
theta_min[phi_i]       = theta_min_ref;
theta_max[phi_i]       = theta_min_ref + theta_delta_ref;
theta_delta[phi_i]     = theta_delta_ref;
inv_theta_delta[phi_i] = 1./theta_delta[phi_i];
theta_dim[phi_i]       = 1;
} else {
phi = 0.5*M_PI - (std::abs((phi_i + 0.5)*phi_delta + phi_min) - absolute_phi_min)  ;
if (std::abs(sin(phi)) > theta_delta_ref / theta_interval) {
theta_delta[phi_i] = std::min(theta_delta_ref / std::abs(sin(phi)),theta_interval);
theta_dim[phi_i]   = std::max((unsigned int)(round(theta_interval / theta_delta[phi_i])), theta_dim_min);
if (accumulation_correction_) {
theta_delta[phi_i] = theta_interval / (theta_dim[phi_i]-1);
theta_min[phi_i]   = theta_min_ref - 0.99*theta_delta[phi_i]*rand_->uniform();
theta_max[phi_i]   = theta_delta[phi_i]*theta_dim[phi_i] + theta_min[phi_i];
} else {
theta_delta[phi_i] = theta_interval / (theta_dim[phi_i]);
theta_min[phi_i]   = theta_min_ref;
theta_max[phi_i]   = theta_delta[phi_i]*theta_dim[phi_i] + theta_min[phi_i];
}
} else {
theta_dim[phi_i]   = theta_dim_min;
if (accumulation_correction_) {
theta_delta[phi_i] = theta_interval / (theta_dim[phi_i]-1);
theta_min[phi_i]   = theta_min_ref - 0.99*theta_delta[phi_i]*rand_->uniform();
theta_max[phi_i]   = theta_delta[phi_i]*theta_dim[phi_i] + theta_min[phi_i];
} else {
theta_delta[phi_i] = theta_interval / (theta_dim[phi_i]);
theta_min[phi_i]   = theta_min_ref;
theta_max[phi_i]   = theta_delta[phi_i]*theta_dim[phi_i] + theta_min[phi_i];
}
}
inv_theta_delta[phi_i] = 1./theta_delta[phi_i];

}


}

unsigned int momentum_cells = 0;
#pragma omp simd reduction(+:momentum_cells)
for(phi_i=0 ; phi_i < phi_dim ; phi_i++) {
momentum_cells += theta_dim[phi_i] * mr_dim;
}

unsigned int momentum_angular_cells = 0;
#pragma omp simd reduction(+:momentum_angular_cells)
for(phi_i=0 ; phi_i < phi_dim ; phi_i++) {
momentum_angular_cells += theta_dim[phi_i];
}

unsigned int  * particles_per_momentum_cells = new unsigned int [momentum_cells];

unsigned int  * momentum_cell_particle_index = new unsigned int [momentum_cells];
#pragma omp simd
for (ic = 0 ; ic < momentum_cells ; ic++) {
momentum_cell_particle_index[ic] = 0;
particles_per_momentum_cells[ic] = 0;
}

unsigned int *  theta_start_index = new unsigned int [phi_dim];

theta_start_index[0] = 0;
for (phi_i = 1 ; phi_i < phi_dim ; phi_i++) {
theta_start_index[phi_i] = theta_start_index[phi_i-1] + theta_dim[phi_i-1];
}


double  * cell_vec_x = new double [momentum_angular_cells];
double  * cell_vec_y = new double [momentum_angular_cells];
double  * cell_vec_z = new double [momentum_angular_cells];

for (phi_i = 0 ; phi_i < phi_dim ; phi_i ++) {

#pragma omp simd private(theta, phi, icc)
for (theta_i = 0 ; theta_i < theta_dim[phi_i] ; theta_i ++) {

icc = theta_start_index[phi_i] + theta_i;

theta = theta_min[phi_i] + (theta_i + 0.5) * theta_delta[phi_i];
phi = phi_min + (phi_i + 0.5) * phi_delta;

cell_vec_x[icc] = cos(phi)*cos(theta);
cell_vec_y[icc] = cos(phi)*sin(theta);
cell_vec_z[icc] = sin(phi);

}
}

if (log_scale_) {
#pragma novector
for (ipr= 0; ipr < number_of_particles ; ipr++ ) {

if (momentum_norm[ipr] > min_momentum_log_scale_)
{
mr_i = (unsigned int) floor( (log10(momentum_norm[ipr]) - mr_min) * inv_mr_delta);
} else {
mr_i = 0;
}
phi_i   = (unsigned int) floor( (particles_phi[ipr] - phi_min) * inv_phi_delta);
theta_i = (unsigned int) floor( (particles_theta[ipr] - theta_min[phi_i]) * inv_theta_delta[phi_i]);

momentum_cell_index[ipr] = (theta_start_index[phi_i]
+ theta_i) * mr_dim + mr_i;


}
} else {
#pragma novector
for (ipr= 0; ipr < number_of_particles ; ipr++ ) {

mr_i    = (unsigned int) floor( (momentum_norm[ipr] - mr_min) * inv_mr_delta);
phi_i   = (unsigned int) floor( (particles_phi[ipr] - phi_min)      * inv_phi_delta);
theta_i = (unsigned int) floor( (particles_theta[ipr] - theta_min[phi_i])  * inv_theta_delta[phi_i]);

momentum_cell_index[ipr] = (theta_start_index[phi_i]
+ theta_i) * mr_dim + mr_i;


}
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





for (phi_i=0 ; phi_i< phi_dim; phi_i++ ) {
for (theta_i=0 ; theta_i< theta_dim[phi_i]; theta_i++ ) {

icc = theta_i + theta_start_index[phi_i] ;

for (mr_i=0 ; mr_i< mr_dim; mr_i++ ) {

ic = mr_i + icc*mr_dim;

if (particles_per_momentum_cells[ic] >= min_packet_size_ ) {

npack = particles_per_momentum_cells[ic]/max_packet_size_;

if (particles_per_momentum_cells[ic]%max_packet_size_ >= min_packet_size_) {
npack += 1;
}

for (ipack = 0 ; ipack < npack ; ipack += 1) {



total_weight = 0;
total_energy = 0;
double total_momentum_x = 0;
double total_momentum_y = 0;
double total_momentum_z = 0;
double total_momentum_norm = 0;

ipr_min = ipack*max_packet_size_;

ipr_max = std::min((ipack+1)*max_packet_size_,particles_per_momentum_cells[ic]);

if (mass == 0) {

for (ipr = ipr_min ; ipr < ipr_max ; ipr ++) {

ip = sorted_particles[momentum_cell_particle_index[ic] + ipr];

total_weight += weight[ip];

total_momentum_x += momentum_x[ip]*weight[ip];
total_momentum_y += momentum_y[ip]*weight[ip];
total_momentum_z += momentum_z[ip]*weight[ip];

total_energy += weight[ip]*momentum_norm[ip - istart];

}

} else {

for (ipr = ipr_min ; ipr < ipr_max ; ipr ++) {

ip = sorted_particles[momentum_cell_particle_index[ic] + ipr];

total_weight += weight[ip];

total_momentum_x += momentum_x[ip]*weight[ip];
total_momentum_y += momentum_y[ip]*weight[ip];
total_momentum_z += momentum_z[ip]*weight[ip];

total_energy += weight[ip]
* sqrt(1.0 + momentum_norm[ip - istart]*momentum_norm[ip - istart]);

}
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


e3_x = e1_y*cell_vec_z[icc] - e1_z*cell_vec_y[icc];
e3_y = e1_z*cell_vec_x[icc] - e1_x*cell_vec_z[icc];
e3_z = e1_x*cell_vec_y[icc] - e1_y*cell_vec_x[icc];

if (std::abs(e3_x*e3_x + e3_y*e3_y + e3_z*e3_z) > 0)
{


e2_x = e1_y*e3_z - e1_z*e3_y;
e2_y = e1_z*e3_x - e1_x*e3_z;
e2_z = e1_x*e3_y - e1_y*e3_x;

e2_norm = sqrt(e2_x*e2_x + e2_y*e2_y + e2_z*e2_z);

e2_x = e2_x / e2_norm;
e2_y = e2_y / e2_norm;
e2_z = e2_z / e2_norm;

ip = sorted_particles[momentum_cell_particle_index[ic] + ipr_min];
momentum_x[ip] = new_momentum_norm*(cos_omega*e1_x + sin_omega*e2_x);
momentum_y[ip] = new_momentum_norm*(cos_omega*e1_y + sin_omega*e2_y);
momentum_z[ip] = new_momentum_norm*(cos_omega*e1_z + sin_omega*e2_z);
weight[ip] = 0.5 * total_weight;

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

if (mass == 0)
{

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


delete [] momentum_norm;
delete [] momentum_cell_index;
delete [] cell_vec_x;
delete [] cell_vec_y;
delete [] cell_vec_z;
delete [] particles_phi;
delete [] particles_theta;
delete [] sorted_particles;
delete [] particles_per_momentum_cells;
delete [] momentum_cell_particle_index;
delete [] theta_start_index;
delete [] theta_dim;
delete [] theta_min;
delete [] theta_max;
delete [] theta_delta;
delete [] inv_theta_delta;

}

}
