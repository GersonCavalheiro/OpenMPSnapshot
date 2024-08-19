

#pragma once
#include "lsqt.h"

class Vector
{
public:
Vector(int n);
Vector(Vector& original);
~Vector();

void add(Vector& other);
void copy(Vector& other);
void apply_sz(Vector& other);
void copy_from_host(real* other_real, real* other_imag);
void copy_to_host(real* target_real, real* target_imag);
void swap(Vector& other);
void inner_product_1(int, Vector& other, Vector& target, int offset);
void inner_product_2(int, int, Vector& target);

#ifndef CPU_ONLY
buffer<real,1> real_part;
buffer<real,1> imag_part;
#else
real* real_part;
real* imag_part;
#endif

private:
void initialize_gpu(int n);
void initialize_cpu(int n);
int n;
int array_size;
};
