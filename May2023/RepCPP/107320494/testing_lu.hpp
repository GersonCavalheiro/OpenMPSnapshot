

#pragma once
#ifndef TESTING_LU_HPP
#define TESTING_LU_HPP

#include "utility.hpp"

#include <rocalution/rocalution.hpp>

using namespace rocalution;

static bool check_residual(float res)
{
return (res < 1e-3f);
}

static bool check_residual(double res)
{
return (res < 1e-6);
}

template <typename T>
bool testing_lu(Arguments argus)
{
int          ndim        = argus.size;
unsigned int format      = argus.format;
std::string  matrix_type = argus.matrix_type;

set_device_rocalution(device);
init_rocalution();

LocalMatrix<T> A;
LocalVector<T> x;
LocalVector<T> b;
LocalVector<T> e;

int* csr_ptr = NULL;
int* csr_col = NULL;
T*   csr_val = NULL;

int nrow = 0;
int ncol = 0;
if(matrix_type == "Laplacian2D")
{
nrow = gen_2d_laplacian(ndim, &csr_ptr, &csr_col, &csr_val);
ncol = nrow;
}
else if(matrix_type == "PermutedIdentity")
{
nrow = gen_permuted_identity(ndim, &csr_ptr, &csr_col, &csr_val);
ncol = nrow;
}
else
{
return false;
}
int nnz = csr_ptr[nrow];

A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

A.MoveToAccelerator();
x.MoveToAccelerator();
b.MoveToAccelerator();
e.MoveToAccelerator();

x.Allocate("x", A.GetN());
b.Allocate("b", A.GetM());
e.Allocate("e", A.GetN());

e.Ones();
A.Apply(e, &b);

x.SetRandomUniform(12345ULL, -4.0, 6.0);

LU<LocalMatrix<T>, LocalVector<T>, T> dls;

dls.Verbose(0);
dls.SetOperator(A);

dls.Build();
dls.Print();

A.ConvertTo(format, format == BCSR ? argus.blockdim : 1);

dls.Solve(b, &x);

x.ScaleAdd(-1.0, e);
T nrm2 = x.Norm();

bool success = check_residual(nrm2);

dls.Clear();

stop_rocalution();

return success;
}

#endif 
