#pragma once

#include <algorithm>

#include <cusp/coo_matrix.h>
#include <cusp/multiply.h>
#include "operator.h"

namespace dg
{




template< class T>
Operator<T> tensorproduct( const Operator<T>& op1, const Operator<T>& op2)
{
#ifdef DG_DEBUG
assert( op1.size() == op2.size());
#endif 
unsigned n = op1.size();
Operator<T> prod( n*n);
for( unsigned i=0; i<n; i++)
for( unsigned j=0; j<n; j++)
for( unsigned k=0; k<n; k++)
for( unsigned l=0; l<n; l++)
prod(i*n+k, j*n+l) = op1(i,j)*op2(k,l);
return prod;
}



template< class T>
cusp::coo_matrix<int,T, cusp::host_memory> tensorproduct( unsigned N, const Operator<T>& op)
{
assert( N>0);
unsigned n = op.size();
unsigned number = n*n;
cusp::coo_matrix<int, T, cusp::host_memory> A(n*N, n*N, N*number);
number = 0;
for( unsigned k=0; k<N; k++)
for( unsigned i=0; i<n; i++)
for( unsigned j=0; j<n; j++)
{
A.row_indices[number]      = k*n+i;
A.column_indices[number]   = k*n+j;
A.values[number]           = op(i,j);
number++;
}
return A;
}



template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> sandwich( const Operator<T>& left,  const cusp::coo_matrix<int, T, cusp::host_memory>& m, const Operator<T>& right)
{
assert( left.size() == right.size());
typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
unsigned n = left.size();
unsigned N = m.num_rows/n;
Matrix r = tensorproduct( N, right);
Matrix l = tensorproduct( N, left);
Matrix mr(m ), lmr(m);

cusp::multiply( m, r, mr);
cusp::multiply( l, mr, lmr);
return lmr;
}





}


