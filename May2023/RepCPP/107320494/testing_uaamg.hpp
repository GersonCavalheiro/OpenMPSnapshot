

#pragma once
#ifndef TESTING_UAAMG_HPP
#define TESTING_UAAMG_HPP

#include "utility.hpp"

#include <rocalution/rocalution.hpp>

using namespace rocalution;

static bool check_residual(float res)
{
return (res < 1e-2f);
}

static bool check_residual(double res)
{
return (res < 1e-5);
}

template <typename T>
bool testing_uaamg(Arguments argus)
{
int          ndim                = argus.size;
int          pre_iter            = argus.pre_smooth;
int          post_iter           = argus.post_smooth;
std::string  smoother            = argus.smoother;
std::string  coarsening_strategy = argus.coarsening_strategy;
std::string  matrix_type         = argus.matrix_type;
unsigned int format              = argus.format;
int          cycle               = argus.cycle;
bool         scaling             = argus.ordering;
bool         rebuildnumeric      = argus.rebuildnumeric;

set_device_rocalution(device);
init_rocalution();

LocalMatrix<T> A;
LocalVector<T> x;
LocalVector<T> b;
LocalVector<T> b2;
LocalVector<T> e;

int* csr_ptr = NULL;
int* csr_col = NULL;
T*   csr_val = NULL;

int nrow = 0;
if(matrix_type == "Laplacian2D")
{
nrow = gen_2d_laplacian(ndim, &csr_ptr, &csr_col, &csr_val);
}
else if(matrix_type == "Laplacian3D")
{
nrow = gen_3d_laplacian(ndim, &csr_ptr, &csr_col, &csr_val);
}
else
{
return false;
}
int nnz = csr_ptr[nrow];

T* csr_val2 = NULL;
if(rebuildnumeric)
{
csr_val2 = new T[nnz];
for(int i = 0; i < nnz; i++)
{
csr_val2[i] = csr_val[i];
}
}

A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

assert(csr_ptr == NULL);
assert(csr_col == NULL);
assert(csr_val == NULL);

A.MoveToAccelerator();
x.MoveToAccelerator();
b.MoveToAccelerator();
b2.MoveToAccelerator();
e.MoveToAccelerator();

x.Allocate("x", A.GetN());
b.Allocate("b", A.GetM());
b2.Allocate("b2", A.GetM());
e.Allocate("e", A.GetN());

e.Ones();
A.Apply(e, &b);

x.SetRandomUniform(12345ULL, -4.0, 6.0);

FCG<LocalMatrix<T>, LocalVector<T>, T> ls;

UAAMG<LocalMatrix<T>, LocalVector<T>, T> p;

p.SetCoarsestLevel(200);
p.SetCycle(cycle);
p.SetOperator(A);
p.SetManualSmoothers(true);
p.SetManualSolver(true);
p.SetScaling(scaling);

if(coarsening_strategy == "Greedy")
{
p.SetCoarseningStrategy(CoarseningStrategy::Greedy);
}
else if(coarsening_strategy == "PMIS")
{
p.SetCoarseningStrategy(CoarseningStrategy::PMIS);
}
else
{
return false;
}

p.SetCouplingStrength(0.005);
p.SetOverInterp(1.2);
p.BuildHierarchy();

int levels = p.GetNumLevels();

FCG<LocalMatrix<T>, LocalVector<T>, T> cgs;
cgs.Verbose(0);

IterativeLinearSolver<LocalMatrix<T>, LocalVector<T>, T>** sm
= new IterativeLinearSolver<LocalMatrix<T>, LocalVector<T>, T>*[levels - 1];

Preconditioner<LocalMatrix<T>, LocalVector<T>, T>** smooth
= new Preconditioner<LocalMatrix<T>, LocalVector<T>, T>*[levels - 1];

for(int i = 0; i < levels - 1; ++i)
{
sm[i] = new FixedPoint<LocalMatrix<T>, LocalVector<T>, T>;

if(smoother == "FSAI")
smooth[i] = new FSAI<LocalMatrix<T>, LocalVector<T>, T>;
else if(smoother == "ILU")
smooth[i] = new ILU<LocalMatrix<T>, LocalVector<T>, T>;
else
return false;

sm[i]->SetPreconditioner(*(smooth[i]));
sm[i]->Verbose(0);
}

p.SetSmoother(sm);
p.SetSolver(cgs);
p.SetSmootherPreIter(pre_iter);
p.SetSmootherPostIter(post_iter);
p.SetOperatorFormat(format, format == BCSR ? argus.blockdim : 1);
p.InitMaxIter(1);
p.Verbose(0);

ls.Verbose(0);
ls.SetOperator(A);
ls.SetPreconditioner(p);

ls.Init(1e-8, 0.0, 1e+8, 10000);
ls.Build();

if(rebuildnumeric)
{
A.UpdateValuesCSR(csr_val2);
delete[] csr_val2;

A.Apply(e, &b2);

ls.ReBuildNumeric();
}

A.ConvertTo(format, format == BCSR ? argus.blockdim : 1);

ls.Solve(rebuildnumeric ? b2 : b, &x);

x.ScaleAdd(-1.0, e);
T nrm2 = x.Norm();

bool success = check_residual(nrm2);

ls.Clear(); 

stop_rocalution();

for(int i = 0; i < levels - 1; ++i)
{
delete smooth[i];
delete sm[i];
}
delete[] smooth;
delete[] sm;

return success;
}

#endif 
