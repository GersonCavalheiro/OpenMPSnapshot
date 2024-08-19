


#pragma once


#include "random.hpp"
#include "matrix.hpp"
#include <tuple>

namespace advscicomp{





template<SizeT N, typename T>
std::tuple<Matrix<N,N,T>, Matrix<N,N,T>> LU(Matrix<N,N,T> const& A)
{
Matrix<N,N,T> L{}, U{};

for (unsigned ii{0}; ii<N; ++ii)
{
for (unsigned jj{ii}; jj<N; ++jj)
{
U[ii][jj] = A[ii][jj];
for (unsigned kk{0}; kk<ii; ++kk)
U[ii][jj] -= L[ii][kk]*U[kk][jj];
}

for (unsigned jj{ii+1}; jj<N; ++jj)
{
L[jj][ii] = A[jj][ii];
for (unsigned kk{0}; kk<ii; ++kk)
L[jj][ii] -= L[jj][kk]*U[kk][ii];
L[jj][ii] = L[jj][ii] / U[ii][ii];
}

}

for (unsigned ii{0}; ii<N; ++ii)
L[ii][ii] = T{1};

return std::make_tuple(L,U);
}


template<SizeT N, typename T>
void LU_InPlace(Matrix<N,N,T> & A)
{
for (unsigned k{0}; k<N; ++k) 
{
for (unsigned m{k}; m<N; ++m) 
for (unsigned j{0}; j<k; ++j)  
A[k][m] -= A[k][j]*A[j][m];  

for (unsigned i{k+1}; i<N; ++i) 
{
for (unsigned j{0}; j<k; ++j) 
A[i][k] -= A[i][j]*A[j][k];

A[i][k] /= A[k][k]; 
}
}
}


template<SizeT N, typename T>
void LU_Col_InPlace(Matrix<N,N,T> & A)
{
for (unsigned k{0}; k<N; ++k) 
{
for (unsigned i{k+1}; i<N; ++i) 
A[i][k] /= A[k][k]; 


for (unsigned j{k+1}; j<N; ++j) 
for (unsigned i{k+1}; i<N; ++i)  
A[i][j] -= A[i][k]*A[k][j];  
}
}

template<SizeT N, typename T>
std::tuple<Matrix<N,N,T>, Matrix<N,N,T>> UnpackToLU(Matrix<N,N,T> const& lu_result)
{
Matrix<N,N,T> L{};
Matrix<N,N,T> U{};

for (unsigned ii{0}; ii<N; ++ii)
for (unsigned jj{ii}; jj<N; ++jj)
U[ii][jj] = lu_result[ii][jj];

for (unsigned ii{0}; ii<N; ++ii)
for (unsigned jj{0}; jj<ii; ++jj)
L[ii][jj] = lu_result[ii][jj];

for (unsigned ii{0}; ii<N; ++ii)
L[ii][ii] = T{1};

return std::make_tuple(L,U);
}



} 

