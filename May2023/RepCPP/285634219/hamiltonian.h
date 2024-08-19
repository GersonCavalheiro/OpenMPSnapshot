

#pragma once
#include "lsqt.h"
class Vector;
class Model;

class Hamiltonian
{
public:
Hamiltonian(Model&);
~Hamiltonian();

void apply(Vector&, Vector&);
void apply_commutator(Vector&, Vector&);
void apply_current(Vector&, Vector&);
void kernel_polynomial(Vector&, Vector&, Vector&);
void chebyshev_01(Vector&, Vector&, Vector&, real, real, int);
void chebyshev_2(Vector&, Vector&, Vector&, Vector&, real, int);
void chebyshev_1x(Vector&, Vector&, real);
void chebyshev_2x(Vector&, Vector&, Vector&, Vector&, Vector&, Vector&, Vector&, real, int);

private:
void initialize_gpu(Model&);
void initialize_cpu(Model&);

#ifndef CPU_ONLY
buffer< int, 1> neighbor_number;
buffer< int, 1> neighbor_list;
buffer<real, 1> potential;
buffer<real, 1> hopping_real;
buffer<real, 1> hopping_imag;
buffer<real, 1> xx;
#else
int* neighbor_number;
int* neighbor_list;
real* potential;
real* hopping_real;
real* hopping_imag;
real* xx;
#endif

int grid_size;
int n;
int max_neighbor;
real energy_max;
};
