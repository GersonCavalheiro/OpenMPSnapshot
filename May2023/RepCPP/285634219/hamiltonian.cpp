

#include "hamiltonian.h"
#include "model.h"
#include "vector.h"
#include <string.h>    
#define BLOCK_SIZE 256

#ifndef CPU_ONLY
void Hamiltonian::initialize_gpu(Model& model)
{
n = model.number_of_atoms;
max_neighbor = model.max_neighbor;
energy_max = model.energy_max;
grid_size = (model.number_of_atoms - 1) / BLOCK_SIZE + 1;

neighbor_number = model.neighbor_number;
#pragma omp target enter data map (to: neighbor_number[0:n])

potential = model.potential;
#pragma omp target enter data map (to: potential[0:n])

int* neighbor_list_new = new int[model.number_of_pairs];
for (int m = 0; m < max_neighbor; ++m) {
for (int i = 0; i < n; ++i) {
neighbor_list_new[m * n + i] = model.neighbor_list[i * max_neighbor + m];
}
}
memcpy(model.neighbor_list, neighbor_list_new, model.number_of_pairs * sizeof(int));
delete[] neighbor_list_new;
neighbor_list = model.neighbor_list;
#pragma omp target enter data map (to: neighbor_list[0:model.number_of_pairs])

real* hopping_real_new = new real[model.number_of_pairs];
for (int m = 0; m < max_neighbor; ++m) {
for (int i = 0; i < n; ++i) {
hopping_real_new[m * n + i] = model.hopping_real[i * max_neighbor + m];
}
}
memcpy(model.hopping_real, hopping_real_new, model.number_of_pairs * sizeof(real));
delete[] hopping_real_new;
hopping_real = model.hopping_real;
#pragma omp target enter data map (to: hopping_real[0:model.number_of_pairs])

real* hopping_imag_new = new real[model.number_of_pairs];
for (int m = 0; m < max_neighbor; ++m) {
for (int i = 0; i < n; ++i) {
hopping_imag_new[m * n + i] = model.hopping_imag[i * max_neighbor + m];
}
}
memcpy(model.hopping_imag, hopping_imag_new, model.number_of_pairs * sizeof(real));
delete[] hopping_imag_new;
hopping_imag = model.hopping_imag;
#pragma omp target enter data map (to: hopping_imag[0:model.number_of_pairs])

real* xx_new = new real[model.number_of_pairs];
for (int m = 0; m < max_neighbor; ++m) {
for (int i = 0; i < n; ++i) {
xx_new[m * n + i] = model.xx[i * max_neighbor + m];
}
}
memcpy(model.xx, xx_new, model.number_of_pairs * sizeof(real));
delete[] xx_new;
xx = model.xx;
#pragma omp target enter data map (to: xx[0:model.number_of_pairs])
}
#else
void Hamiltonian::initialize_cpu(Model& model)
{
n = model.number_of_atoms;
max_neighbor = model.max_neighbor;
energy_max = model.energy_max;
int number_of_pairs = model.number_of_pairs;

neighbor_number = new int[n];
memcpy(neighbor_number, model.neighbor_number, sizeof(int) * n);
delete[] model.neighbor_number;

neighbor_list = new int[number_of_pairs];
memcpy(neighbor_list, model.neighbor_list, sizeof(int) * number_of_pairs);
delete[] model.neighbor_list;

potential = new real[n];
memcpy(potential, model.potential, sizeof(real) * n);
delete[] model.potential;

hopping_real = new real[number_of_pairs];
memcpy(hopping_real, model.hopping_real, sizeof(real) * number_of_pairs);
delete[] model.hopping_real;

hopping_imag = new real[number_of_pairs];
memcpy(hopping_imag, model.hopping_imag, sizeof(real) * number_of_pairs);
delete[] model.hopping_imag;

xx = new real[number_of_pairs];
memcpy(xx, model.xx, sizeof(real) * number_of_pairs);
delete[] model.xx;
}
#endif

Hamiltonian::Hamiltonian(Model& model)
{
#ifndef CPU_ONLY
initialize_gpu(model);
#else
initialize_cpu(model);
#endif
}

Hamiltonian::~Hamiltonian()
{
#ifndef CPU_ONLY

#pragma omp target exit data map(delete: neighbor_number[0:n])
#pragma omp target exit data map(delete: neighbor_list[0:max_neighbor*n])
#pragma omp target exit data map(delete: potential[0:n])
#pragma omp target exit data map(delete: hopping_real[0:max_neighbor*n])
#pragma omp target exit data map(delete: hopping_imag[0:max_neighbor*n])
#pragma omp target exit data map(delete: xx[0:max_neighbor*n])
delete[] neighbor_number;
delete[] neighbor_list;
delete[] potential;
delete[] hopping_real;
delete[] hopping_imag;
delete[] xx;
#else
delete[] neighbor_number;
delete[] neighbor_list;
delete[] potential;
delete[] hopping_real;
delete[] hopping_imag;
delete[] xx;
#endif
}

#ifndef CPU_ONLY
void gpu_apply_hamiltonian(
const int number_of_atoms,
const real energy_max,
const  int* __restrict g_neighbor_number,
const  int* __restrict g_neighbor_list,
const real* __restrict g_potential,
const real* __restrict g_hopping_real,
const real* __restrict g_hopping_imag,
const real* __restrict g_state_in_real,
const real* __restrict g_state_in_imag,
real* __restrict g_state_out_real,
real* __restrict g_state_out_imag)
{
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
for (int n = 0; n < number_of_atoms; n++) {
real temp_real = g_potential[n] * g_state_in_real[n]; 
real temp_imag = g_potential[n] * g_state_in_imag[n]; 

for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = m * number_of_atoms + n;
int index_2 = g_neighbor_list[index_1];
real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_in_real[index_2];
real d = g_state_in_imag[index_2];
temp_real += a * c - b * d; 
temp_imag += a * d + b * c; 
}
temp_real /= energy_max; 
temp_imag /= energy_max; 
g_state_out_real[n] = temp_real;
g_state_out_imag[n] = temp_imag;
}
}
#else
void cpu_apply_hamiltonian(
int number_of_atoms,
int max_neighbor,
real energy_max,
int* g_neighbor_number,
int* g_neighbor_list,
real* g_potential,
real* g_hopping_real,
real* g_hopping_imag,
real* g_state_in_real,
real* g_state_in_imag,
real* g_state_out_real,
real* g_state_out_imag)
{
for (int n = 0; n < number_of_atoms; ++n) {
real temp_real = g_potential[n] * g_state_in_real[n]; 
real temp_imag = g_potential[n] * g_state_in_imag[n]; 

for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = n * max_neighbor + m;
int index_2 = g_neighbor_list[index_1];
real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_in_real[index_2];
real d = g_state_in_imag[index_2];
temp_real += a * c - b * d; 
temp_imag += a * d + b * c; 
}
temp_real /= energy_max; 
temp_imag /= energy_max; 
g_state_out_real[n] = temp_real;
g_state_out_imag[n] = temp_imag;
}
}
#endif

void Hamiltonian::apply(Vector& input, Vector& output)
{
#ifndef CPU_ONLY
gpu_apply_hamiltonian(
n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag,
input.real_part, input.imag_part, output.real_part, output.imag_part);
#else
cpu_apply_hamiltonian(
n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
hopping_imag, input.real_part, input.imag_part, output.real_part, output.imag_part);
#endif
}

#ifndef CPU_ONLY
void gpu_apply_commutator(
int number_of_atoms,
real energy_max,
int* g_neighbor_number,
int* g_neighbor_list,
real* g_hopping_real,
real* g_hopping_imag,
real* g_xx,
real* g_state_in_real,
real* g_state_in_imag,
real* g_state_out_real,
real* g_state_out_imag)
{
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
for (int n = 0; n < number_of_atoms; n++) {
real temp_real = 0.0;
real temp_imag = 0.0;
for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = m * number_of_atoms + n;
int index_2 = g_neighbor_list[index_1];
real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_in_real[index_2];
real d = g_state_in_imag[index_2];
real xx = g_xx[index_1];
temp_real -= (a * c - b * d) * xx;
temp_imag -= (a * d + b * c) * xx;
}
g_state_out_real[n] = temp_real / energy_max; 
g_state_out_imag[n] = temp_imag / energy_max; 
}
}
#else
void cpu_apply_commutator(
int number_of_atoms,
int max_neighbor,
real energy_max,
int* g_neighbor_number,
int* g_neighbor_list,
real* g_hopping_real,
real* g_hopping_imag,
real* g_xx,
real* g_state_in_real,
real* g_state_in_imag,
real* g_state_out_real,
real* g_state_out_imag)
{
for (int n = 0; n < number_of_atoms; ++n) {
real temp_real = 0.0;
real temp_imag = 0.0;
for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = n * max_neighbor + m;
int index_2 = g_neighbor_list[index_1];
real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_in_real[index_2];
real d = g_state_in_imag[index_2];
real xx = g_xx[index_1];
temp_real -= (a * c - b * d) * xx;
temp_imag -= (a * d + b * c) * xx;
}
g_state_out_real[n] = temp_real / energy_max; 
g_state_out_imag[n] = temp_imag / energy_max; 
}
}
#endif

void Hamiltonian::apply_commutator(Vector& input, Vector& output)
{
#ifndef CPU_ONLY
gpu_apply_commutator(
n, energy_max, neighbor_number, neighbor_list, hopping_real, hopping_imag, xx, input.real_part,
input.imag_part, output.real_part, output.imag_part);
#else
cpu_apply_commutator(
n, max_neighbor, energy_max, neighbor_number, neighbor_list, hopping_real, hopping_imag, xx,
input.real_part, input.imag_part, output.real_part, output.imag_part);
#endif
}

#ifndef CPU_ONLY
void gpu_apply_current(
const int number_of_atoms,
const  int* __restrict g_neighbor_number,
const  int* __restrict g_neighbor_list,
const real* __restrict g_hopping_real,
const real* __restrict g_hopping_imag,
const real* __restrict g_xx,
const real* __restrict g_state_in_real,
const real* __restrict g_state_in_imag,
real* __restrict g_state_out_real,
real* __restrict g_state_out_imag)
{
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
for (int n = 0; n < number_of_atoms; n++) {
real temp_real = 0.0;
real temp_imag = 0.0;
for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = m * number_of_atoms + n;
int index_2 = g_neighbor_list[index_1];
real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_in_real[index_2];
real d = g_state_in_imag[index_2];
temp_real += (a * c - b * d) * g_xx[index_1];
temp_imag += (a * d + b * c) * g_xx[index_1];
}
g_state_out_real[n] = +temp_imag;
g_state_out_imag[n] = -temp_real;
}
}
#else
void cpu_apply_current(
int number_of_atoms,
int max_neighbor,
int* g_neighbor_number,
int* g_neighbor_list,
real* g_hopping_real,
real* g_hopping_imag,
real* g_xx,
real* g_state_in_real,
real* g_state_in_imag,
real* g_state_out_real,
real* g_state_out_imag)
{
for (int n = 0; n < number_of_atoms; ++n) {
real temp_real = 0.0;
real temp_imag = 0.0;
for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = n * max_neighbor + m;
int index_2 = g_neighbor_list[index_1];
real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_in_real[index_2];
real d = g_state_in_imag[index_2];
temp_real += (a * c - b * d) * g_xx[index_1];
temp_imag += (a * d + b * c) * g_xx[index_1];
}
g_state_out_real[n] = +temp_imag;
g_state_out_imag[n] = -temp_real;
}
}
#endif

void Hamiltonian::apply_current(Vector& input, Vector& output)
{
#ifndef CPU_ONLY
gpu_apply_current(
n, neighbor_number, neighbor_list, hopping_real, hopping_imag, xx, input.real_part,
input.imag_part, output.real_part, output.imag_part);
#else
cpu_apply_current(
n, max_neighbor, neighbor_number, neighbor_list, hopping_real, hopping_imag, xx,
input.real_part, input.imag_part, output.real_part, output.imag_part);
#endif
}

#ifndef CPU_ONLY
void gpu_chebyshev_01(
const int number_of_atoms,
const real* __restrict g_state_0_real,
const real* __restrict g_state_0_imag,
const real* __restrict g_state_1_real,
const real* __restrict g_state_1_imag,
real* __restrict g_state_real,
real* __restrict g_state_imag,
const real b0,
const real b1,
const int direction)
{
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
for (int n = 0; n < number_of_atoms; n++) {
real bessel_0 = b0;
real bessel_1 = b1 * direction;
g_state_real[n] = bessel_0 * g_state_0_real[n] + bessel_1 * g_state_1_imag[n];
g_state_imag[n] = bessel_0 * g_state_0_imag[n] - bessel_1 * g_state_1_real[n];
}
}
#else
void cpu_chebyshev_01(
int number_of_atoms,
real* g_state_0_real,
real* g_state_0_imag,
real* g_state_1_real,
real* g_state_1_imag,
real* g_state_real,
real* g_state_imag,
real b0,
real b1,
int direction)
{
for (int n = 0; n < number_of_atoms; ++n) {
real bessel_0 = b0;
real bessel_1 = b1 * direction;
g_state_real[n] = bessel_0 * g_state_0_real[n] + bessel_1 * g_state_1_imag[n];
g_state_imag[n] = bessel_0 * g_state_0_imag[n] - bessel_1 * g_state_1_real[n];
}
}
#endif

void Hamiltonian::chebyshev_01(
Vector& state_0, Vector& state_1, Vector& state, real bessel_0, real bessel_1, int direction)
{
#ifndef CPU_ONLY
gpu_chebyshev_01(
n, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state.real_part,
state.imag_part, bessel_0, bessel_1, direction);
#else
cpu_chebyshev_01(
n, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state.real_part,
state.imag_part, bessel_0, bessel_1, direction);
#endif
}

#ifndef CPU_ONLY
void gpu_chebyshev_2(
const int number_of_atoms,
const real energy_max,
const  int* __restrict g_neighbor_number,
const  int* __restrict g_neighbor_list,
const real* __restrict g_potential,
const real* __restrict g_hopping_real,
const real* __restrict g_hopping_imag,
const real* __restrict g_state_0_real,
const real* __restrict g_state_0_imag,
const real* __restrict g_state_1_real,
const real* __restrict g_state_1_imag,
real* __restrict g_state_2_real,
real* __restrict g_state_2_imag,
real* __restrict g_state_real,
real* __restrict g_state_imag,
const real bessel_m,
const int label)
{
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
for (int n = 0; n < number_of_atoms; n++) {
real temp_real = g_potential[n] * g_state_1_real[n]; 
real temp_imag = g_potential[n] * g_state_1_imag[n]; 

for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = m * number_of_atoms + n;
int index_2 = g_neighbor_list[index_1];
real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_1_real[index_2];
real d = g_state_1_imag[index_2];
temp_real += a * c - b * d; 
temp_imag += a * d + b * c; 
}
temp_real /= energy_max; 
temp_imag /= energy_max; 

temp_real = 2.0 * temp_real - g_state_0_real[n];
temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
switch (label) {
case 1: {
g_state_real[n] += bessel_m * temp_real;
g_state_imag[n] += bessel_m * temp_imag;
break;
}
case 2: {
g_state_real[n] -= bessel_m * temp_real;
g_state_imag[n] -= bessel_m * temp_imag;
break;
}
case 3: {
g_state_real[n] += bessel_m * temp_imag;
g_state_imag[n] -= bessel_m * temp_real;
break;
}
case 4: {
g_state_real[n] -= bessel_m * temp_imag;
g_state_imag[n] += bessel_m * temp_real;
break;
}
}
g_state_2_real[n] = temp_real;
g_state_2_imag[n] = temp_imag;
}
}
#else
void cpu_chebyshev_2(
int number_of_atoms,
int max_neighbor,
real energy_max,
int* g_neighbor_number,
int* g_neighbor_list,
real* g_potential,
real* g_hopping_real,
real* g_hopping_imag,
real* g_state_0_real,
real* g_state_0_imag,
real* g_state_1_real,
real* g_state_1_imag,
real* g_state_2_real,
real* g_state_2_imag,
real* g_state_real,
real* g_state_imag,
real bessel_m,
int label)
{
for (int n = 0; n < number_of_atoms; ++n) {
real temp_real = g_potential[n] * g_state_1_real[n]; 
real temp_imag = g_potential[n] * g_state_1_imag[n]; 

for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = n * max_neighbor + m;
int index_2 = g_neighbor_list[index_1];
real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_1_real[index_2];
real d = g_state_1_imag[index_2];
temp_real += a * c - b * d; 
temp_imag += a * d + b * c; 
}
temp_real /= energy_max; 
temp_imag /= energy_max; 

temp_real = 2.0 * temp_real - g_state_0_real[n];
temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
switch (label) {
case 1: {
g_state_real[n] += bessel_m * temp_real;
g_state_imag[n] += bessel_m * temp_imag;
break;
}
case 2: {
g_state_real[n] -= bessel_m * temp_real;
g_state_imag[n] -= bessel_m * temp_imag;
break;
}
case 3: {
g_state_real[n] += bessel_m * temp_imag;
g_state_imag[n] -= bessel_m * temp_real;
break;
}
case 4: {
g_state_real[n] -= bessel_m * temp_imag;
g_state_imag[n] += bessel_m * temp_real;
break;
}
}
g_state_2_real[n] = temp_real;
g_state_2_imag[n] = temp_imag;
}
}
#endif

void Hamiltonian::chebyshev_2(
Vector& state_0, Vector& state_1, Vector& state_2, Vector& state, real bessel_m, int label)
{
#ifndef CPU_ONLY
gpu_chebyshev_2(
n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag,
state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state_2.real_part,
state_2.imag_part, state.real_part, state.imag_part, bessel_m, label);
#else
cpu_chebyshev_2(
n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
hopping_imag, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part,
state_2.real_part, state_2.imag_part, state.real_part, state.imag_part, bessel_m, label);
#endif
}

#ifndef CPU_ONLY
void gpu_chebyshev_1x(
const int number_of_atoms,
const real* __restrict g_state_1x_real,
const real* __restrict g_state_1x_imag,
real* __restrict g_state_real,
real* __restrict g_state_imag,
const real g_bessel_1)
{
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
for (int n = 0; n < number_of_atoms; n++) {
real b1 = g_bessel_1;
g_state_real[n] = +b1 * g_state_1x_imag[n];
g_state_imag[n] = -b1 * g_state_1x_real[n];
}
}
#else
void cpu_chebyshev_1x(
int number_of_atoms,
real* g_state_1x_real,
real* g_state_1x_imag,
real* g_state_real,
real* g_state_imag,
real g_bessel_1)
{
for (int n = 0; n < number_of_atoms; ++n) {
real b1 = g_bessel_1;
g_state_real[n] = +b1 * g_state_1x_imag[n];
g_state_imag[n] = -b1 * g_state_1x_real[n];
}
}
#endif

void Hamiltonian::chebyshev_1x(Vector& input, Vector& output, real bessel_1)
{
#ifndef CPU_ONLY
gpu_chebyshev_1x(
n, input.real_part, input.imag_part, output.real_part, output.imag_part, bessel_1);
#else
cpu_chebyshev_1x(
n, input.real_part, input.imag_part, output.real_part, output.imag_part, bessel_1);
#endif
}

#ifndef CPU_ONLY
void gpu_chebyshev_2x(
const int number_of_atoms,
const real energy_max,
const  int* __restrict g_neighbor_number,
const  int* __restrict g_neighbor_list,
const real* __restrict g_potential,
const real* __restrict g_hopping_real,
const real* __restrict g_hopping_imag,
const real* __restrict g_xx,
const real* __restrict g_state_0_real,
const real* __restrict g_state_0_imag,
const real* __restrict g_state_0x_real,
const real* __restrict g_state_0x_imag,
const real* __restrict g_state_1_real,
const real* __restrict g_state_1_imag,
const real* __restrict g_state_1x_real,
const real* __restrict g_state_1x_imag,
real* __restrict g_state_2_real,
real* __restrict g_state_2_imag,
real* __restrict g_state_2x_real,
real* __restrict g_state_2x_imag,
real* __restrict g_state_real,
real* __restrict g_state_imag,
const real g_bessel_m,
const int g_label)
{
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
for (int n = 0; n < number_of_atoms; n++) {
real temp_real = g_potential[n] * g_state_1_real[n];    
real temp_imag = g_potential[n] * g_state_1_imag[n];    
real temp_x_real = g_potential[n] * g_state_1x_real[n]; 
real temp_x_imag = g_potential[n] * g_state_1x_imag[n]; 

for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = m * number_of_atoms + n;
int index_2 = g_neighbor_list[index_1];

real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_1_real[index_2];
real d = g_state_1_imag[index_2];
temp_real += a * c - b * d; 
temp_imag += a * d + b * c; 

real cx = g_state_1x_real[index_2];
real dx = g_state_1x_imag[index_2];
temp_x_real += a * cx - b * dx; 
temp_x_imag += a * dx + b * cx; 

real xx = g_xx[index_1];
temp_x_real -= (a * c - b * d) * xx; 
temp_x_imag -= (a * d + b * c) * xx; 
}

temp_real /= energy_max; 
temp_imag /= energy_max; 
temp_real = 2.0 * temp_real - g_state_0_real[n];
temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
g_state_2_real[n] = temp_real;
g_state_2_imag[n] = temp_imag;

temp_x_real /= energy_max; 
temp_x_imag /= energy_max; 
temp_x_real = 2.0 * temp_x_real - g_state_0x_real[n];
temp_x_imag = 2.0 * temp_x_imag - g_state_0x_imag[n];
g_state_2x_real[n] = temp_x_real;
g_state_2x_imag[n] = temp_x_imag;

real bessel_m = g_bessel_m;
switch (g_label) {
case 1: {
g_state_real[n] += bessel_m * temp_x_real;
g_state_imag[n] += bessel_m * temp_x_imag;
break;
}
case 2: {
g_state_real[n] -= bessel_m * temp_x_real;
g_state_imag[n] -= bessel_m * temp_x_imag;
break;
}
case 3: {
g_state_real[n] += bessel_m * temp_x_imag;
g_state_imag[n] -= bessel_m * temp_x_real;
break;
}
case 4: {
g_state_real[n] -= bessel_m * temp_x_imag;
g_state_imag[n] += bessel_m * temp_x_real;
break;
}
}
}
}
#else
void cpu_chebyshev_2x(
int number_of_atoms,
int max_neighbor,
real energy_max,
int* g_neighbor_number,
int* g_neighbor_list,
real* g_potential,
real* g_hopping_real,
real* g_hopping_imag,
real* g_xx,
real* g_state_0_real,
real* g_state_0_imag,
real* g_state_0x_real,
real* g_state_0x_imag,
real* g_state_1_real,
real* g_state_1_imag,
real* g_state_1x_real,
real* g_state_1x_imag,
real* g_state_2_real,
real* g_state_2_imag,
real* g_state_2x_real,
real* g_state_2x_imag,
real* g_state_real,
real* g_state_imag,
real g_bessel_m,
int g_label)
{
for (int n = 0; n < number_of_atoms; ++n) {
real temp_real = g_potential[n] * g_state_1_real[n];    
real temp_imag = g_potential[n] * g_state_1_imag[n];    
real temp_x_real = g_potential[n] * g_state_1x_real[n]; 
real temp_x_imag = g_potential[n] * g_state_1x_imag[n]; 

for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = n * max_neighbor + m;
int index_2 = g_neighbor_list[index_1];

real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_1_real[index_2];
real d = g_state_1_imag[index_2];
temp_real += a * c - b * d; 
temp_imag += a * d + b * c; 

real cx = g_state_1x_real[index_2];
real dx = g_state_1x_imag[index_2];
temp_x_real += a * cx - b * dx; 
temp_x_imag += a * dx + b * cx; 

real xx = g_xx[index_1];
temp_x_real -= (a * c - b * d) * xx; 
temp_x_imag -= (a * d + b * c) * xx; 
}

temp_real /= energy_max; 
temp_imag /= energy_max; 
temp_real = 2.0 * temp_real - g_state_0_real[n];
temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
g_state_2_real[n] = temp_real;
g_state_2_imag[n] = temp_imag;

temp_x_real /= energy_max; 
temp_x_imag /= energy_max; 
temp_x_real = 2.0 * temp_x_real - g_state_0x_real[n];
temp_x_imag = 2.0 * temp_x_imag - g_state_0x_imag[n];
g_state_2x_real[n] = temp_x_real;
g_state_2x_imag[n] = temp_x_imag;

real bessel_m = g_bessel_m;
switch (g_label) {
case 1: {
g_state_real[n] += bessel_m * temp_x_real;
g_state_imag[n] += bessel_m * temp_x_imag;
break;
}
case 2: {
g_state_real[n] -= bessel_m * temp_x_real;
g_state_imag[n] -= bessel_m * temp_x_imag;
break;
}
case 3: {
g_state_real[n] += bessel_m * temp_x_imag;
g_state_imag[n] -= bessel_m * temp_x_real;
break;
}
case 4: {
g_state_real[n] -= bessel_m * temp_x_imag;
g_state_imag[n] += bessel_m * temp_x_real;
break;
}
}
}
}
#endif

void Hamiltonian::chebyshev_2x(
Vector& state_0,
Vector& state_0x,
Vector& state_1,
Vector& state_1x,
Vector& state_2,
Vector& state_2x,
Vector& state,
real bessel_m,
int label)
{
#ifndef CPU_ONLY
gpu_chebyshev_2x(
n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag, xx,
state_0.real_part, state_0.imag_part, state_0x.real_part, state_0x.imag_part, state_1.real_part,
state_1.imag_part, state_1x.real_part, state_1x.imag_part, state_2.real_part, state_2.imag_part,
state_2x.real_part, state_2x.imag_part, state.real_part, state.imag_part, bessel_m, label);
#else
cpu_chebyshev_2x(
n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
hopping_imag, xx, state_0.real_part, state_0.imag_part, state_0x.real_part, state_0x.imag_part,
state_1.real_part, state_1.imag_part, state_1x.real_part, state_1x.imag_part, state_2.real_part,
state_2.imag_part, state_2x.real_part, state_2x.imag_part, state.real_part, state.imag_part,
bessel_m, label);
#endif
}

#ifndef CPU_ONLY
void gpu_kernel_polynomial(
const int number_of_atoms,
const real energy_max,
const  int* __restrict g_neighbor_number,
const  int* __restrict g_neighbor_list,
const real* __restrict g_potential,
const real* __restrict g_hopping_real,
const real* __restrict g_hopping_imag,
const real* __restrict g_state_0_real,
const real* __restrict g_state_0_imag,
const real* __restrict g_state_1_real,
const real* __restrict g_state_1_imag,
real* __restrict g_state_2_real,
real* __restrict g_state_2_imag)
{
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
for (int n = 0; n < number_of_atoms; n++) {
real temp_real = g_potential[n] * g_state_1_real[n]; 
real temp_imag = g_potential[n] * g_state_1_imag[n]; 

for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = m * number_of_atoms + n;
int index_2 = g_neighbor_list[index_1];
real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_1_real[index_2];
real d = g_state_1_imag[index_2];
temp_real += a * c - b * d; 
temp_imag += a * d + b * c; 
}

temp_real /= energy_max; 
temp_imag /= energy_max; 

temp_real = 2.0 * temp_real - g_state_0_real[n];
temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
g_state_2_real[n] = temp_real;
g_state_2_imag[n] = temp_imag;
}
}
#else
void cpu_kernel_polynomial(
int number_of_atoms,
int max_neighbor,
real energy_max,
int* g_neighbor_number,
int* g_neighbor_list,
real* g_potential,
real* g_hopping_real,
real* g_hopping_imag,
real* g_state_0_real,
real* g_state_0_imag,
real* g_state_1_real,
real* g_state_1_imag,
real* g_state_2_real,
real* g_state_2_imag)
{
for (int n = 0; n < number_of_atoms; ++n) {
real temp_real = g_potential[n] * g_state_1_real[n]; 
real temp_imag = g_potential[n] * g_state_1_imag[n]; 

for (int m = 0; m < g_neighbor_number[n]; ++m) {
int index_1 = n * max_neighbor + m;
int index_2 = g_neighbor_list[index_1];
real a = g_hopping_real[index_1];
real b = g_hopping_imag[index_1];
real c = g_state_1_real[index_2];
real d = g_state_1_imag[index_2];
temp_real += a * c - b * d; 
temp_imag += a * d + b * c; 
}

temp_real /= energy_max; 
temp_imag /= energy_max; 

temp_real = 2.0 * temp_real - g_state_0_real[n];
temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
g_state_2_real[n] = temp_real;
g_state_2_imag[n] = temp_imag;
}
}
#endif

void Hamiltonian::kernel_polynomial(Vector& state_0, Vector& state_1, Vector& state_2)
{
#ifndef CPU_ONLY
gpu_kernel_polynomial(
n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag,
state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state_2.real_part,
state_2.imag_part);
#else
cpu_kernel_polynomial(
n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
hopping_imag, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part,
state_2.real_part, state_2.imag_part);
#endif
}
