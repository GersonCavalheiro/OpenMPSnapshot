#include <omp.h>
#include "config.h"


namespace dg{

template<class value_type>
void ell_multiply_kernel( value_type alpha, value_type beta,
const value_type * RESTRICT data, const int * RESTRICT cols_idx,
const int * RESTRICT data_idx,
const int num_rows, const int num_cols, const int blocks_per_line,
const int n,
const int left_size, const int right_size,
const int * RESTRICT right_range,
const value_type * RESTRICT x, value_type * RESTRICT y
)
{
#pragma omp for nowait 
for( int si = 0; si<left_size*num_rows; si++)
{
int s = si / num_rows;
int i = si % num_rows;
#ifdef _MSC_VER 
int* J = (int*)alloca(blocks_per_line * sizeof(int));
#else
int J[blocks_per_line];
#endif
for( int d=0; d<blocks_per_line; d++)
J[d] = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
for( int k=0; k<n; k++)
{
#ifdef _MSC_VER
int* B = (int*)alloca(blocks_per_line * sizeof(int));
#else
int B[blocks_per_line];
#endif
for( int d=0; d<blocks_per_line; d++)
B[d] = (data_idx[i*blocks_per_line+d]*n+k)*n;
for( int j=right_range[0]; j<right_range[1]; j++)
{
int I = ((s*num_rows + i)*n+k)*right_size+j;
y[I] = beta == 0 ? (value_type)0 : y[I]*beta;
for( int d=0; d<blocks_per_line; d++)
{
value_type temp = 0;
for( int q=0; q<n; q++) 
temp = DG_FMA(data[ B[d]+q],
x[(J[d]+q)*right_size+j],
temp);
y[I] = DG_FMA(alpha, temp, y[I]);
}
}
}
}
}
template<class value_type, int n, int blocks_per_line>
void ell_multiply_kernel( value_type alpha, value_type beta,
const value_type * RESTRICT data, const int * RESTRICT cols_idx,
const int * RESTRICT data_idx,
const int num_rows, const int num_cols,
const int left_size, const int right_size,
const int * RESTRICT right_range,
const value_type * RESTRICT x, value_type * RESTRICT y
)
{
if(right_size==1)
{
bool trivial = true;
for( int i=1; i<num_rows-1; i++)
for( int d=0; d<blocks_per_line; d++)
{
if( data_idx[i*blocks_per_line+d]
!= data_idx[blocks_per_line+d]) trivial = false;
}
if(trivial)
{
value_type xprivate[blocks_per_line*n];
value_type dprivate[blocks_per_line*n*n];
for( int d=0; d<blocks_per_line; d++)
for( int k=0; k<n; k++)
for( int q=0; q<n; q++)
{
int B = data_idx[blocks_per_line+d];
dprivate[(k*blocks_per_line+d)*n+q] = data[(B*n+k)*n+q];
}
#pragma omp for nowait
for( int s=0; s<left_size; s++)
{
for( int i=0; i<1; i++)
{
for( int d=0; d<blocks_per_line; d++)
{
int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
for(int q=0; q<n; q++)
xprivate[d*n+q] = x[J+q];
}
for( int k=0; k<n; k++)
{
value_type temp[blocks_per_line] = {0};
for( int d=0; d<blocks_per_line; d++)
{
int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
for( int q=0; q<n; q++) 
temp[d] = DG_FMA(data[B+q], xprivate[d*n+q], temp[d]);
}
int I = ((s*num_rows + i)*n+k);
y[I] = beta == 0 ? (value_type)0 : y[I]*beta;
for( int d=0; d<blocks_per_line; d++)
y[I] = DG_FMA(alpha, temp[d], y[I]);
}
}
#ifndef _MSC_VER
#pragma omp SIMD 
#endif
for( int i=1; i<num_rows-1; i++)
{
for( int k=0; k<n; k++)
{
int I = ((s*num_rows + i)*n+k);
y[I] = beta == 0 ? (value_type)0 : y[I]*beta;
int B = n*blocks_per_line*k;
for( int d=0; d<blocks_per_line; d++)
{
value_type temp = 0;
for( int q=0; q<n; q++)
{
int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n+q;
temp = DG_FMA( dprivate[B+d*n+q], x[J], temp);
}
y[I] = DG_FMA(alpha, temp, y[I]);
}
}
}
for( int i=num_rows-1; i<num_rows; i++)
{
for( int d=0; d<blocks_per_line; d++)
{
int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
for(int q=0; q<n; q++)
xprivate[d*n+q] = x[J+q];
}
for( int k=0; k<n; k++)
{
value_type temp[blocks_per_line] = {0};
for( int d=0; d<blocks_per_line; d++)
{
int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
for( int q=0; q<n; q++) 
temp[d] = DG_FMA( data[B+q], xprivate[d*n+q], temp[d]);
}
int I = ((s*num_rows + i)*n+k);
y[I] = beta == 0 ? (value_type)0 : y[I]*beta;
for( int d=0; d<blocks_per_line; d++)
y[I] = DG_FMA(alpha, temp[d], y[I]);
}
}
}
} 
else 
{
value_type xprivate[blocks_per_line*n];
#pragma omp for nowait
for( int s=0; s<left_size; s++)
for( int i=0; i<num_rows; i++)
{
for( int d=0; d<blocks_per_line; d++)
{
int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
for(int q=0; q<n; q++)
xprivate[d*n+q] = x[J+q];
}
for( int k=0; k<n; k++)
{
value_type temp[blocks_per_line] = {0};
for( int d=0; d<blocks_per_line; d++)
{
int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
for( int q=0; q<n; q++) 
temp[d] = DG_FMA( data[B+q], xprivate[d*n+q], temp[d]);
}
int I = ((s*num_rows + i)*n+k);
y[I] = beta == 0 ? (value_type)0 : y[I]*beta;
for( int d=0; d<blocks_per_line; d++)
y[I] = DG_FMA(alpha, temp[d], y[I]);
}
}
}
}
else 
{
value_type dprivate[blocks_per_line*n];
int J[blocks_per_line];
if( !( (right_range[1]-right_range[0]) > 100*left_size*num_rows*n )) 
{
#pragma omp for nowait
for (int sik = 0; sik < left_size*num_rows*n; sik++)
{
int s = sik / (num_rows*n);
int i = (sik % (num_rows*n)) / n;
int k = (sik % (num_rows*n)) % n;

for( int d=0; d<blocks_per_line; d++)
{
J[d] = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
for(int q=0; q<n; q++)
dprivate[d*n+q] = data[B+q];
}
#ifndef _MSC_VER
#pragma omp SIMD 
#endif
for( int j=right_range[0]; j<right_range[1]; j++)
{
int I = ((s*num_rows + i)*n+k)*right_size+j;
y[I] = beta == 0 ? (value_type)0 : y[I]*beta;
for( int d=0; d<blocks_per_line; d++)
{
value_type temp = 0;
int Jd = J[d];
for( int q=0; q<n; q++) 
temp = DG_FMA( dprivate[ d*n+q],
x[(Jd+q)*right_size+j],
temp);
y[I] = DG_FMA(alpha, temp, y[I]);
}
}
}
}
else 
{

for (int sik = 0; sik < left_size*num_rows*n; sik++)
{
int s = sik / (num_rows*n);
int i = (sik % (num_rows*n)) / n;
int k = (sik % (num_rows*n)) % n;

for( int d=0; d<blocks_per_line; d++)
{
J[d] = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
for(int q=0; q<n; q++)
dprivate[d*n+q] = data[B+q];
}
#pragma omp for SIMD nowait
for( int j=right_range[0]; j<right_range[1]; j++)
{
int I = ((s*num_rows + i)*n+k)*right_size+j;
y[I] = beta == 0 ? (value_type)0 : y[I]*beta;
for( int d=0; d<blocks_per_line; d++)
{
value_type temp = 0;
int Jd = J[d];
for( int q=0; q<n; q++) 
temp = DG_FMA( dprivate[ d*n+q],
x[(Jd+q)*right_size+j],
temp);
y[I] = DG_FMA(alpha, temp, y[I]);
}
}
}
}
}
}

template<class value_type, int n>
void call_ell_multiply_kernel( value_type alpha, value_type beta,
const value_type * RESTRICT data_ptr, const int * RESTRICT cols_ptr,
const int * RESTRICT block_ptr,
const int num_rows, const int num_cols, const int blocks_per_line,
const int left_size, const int right_size,
const int * RESTRICT right_range_ptr,
const value_type * RESTRICT x_ptr, value_type * RESTRICT y_ptr)
{
if( blocks_per_line == 1)
ell_multiply_kernel<value_type, n, 1>  (alpha, beta, data_ptr,
cols_ptr, block_ptr, num_rows, num_cols, left_size, right_size,
right_range_ptr,  x_ptr,y_ptr);
else if (blocks_per_line == 2)
ell_multiply_kernel<value_type, n, 2>  (alpha, beta, data_ptr,
cols_ptr, block_ptr, num_rows, num_cols, left_size, right_size,
right_range_ptr,  x_ptr,y_ptr);
else if (blocks_per_line == 3)
ell_multiply_kernel<value_type, n, 3>  (alpha, beta, data_ptr,
cols_ptr, block_ptr, num_rows, num_cols, left_size, right_size,
right_range_ptr,  x_ptr,y_ptr);
else if (blocks_per_line == 4)
ell_multiply_kernel<value_type, n, 4>  (alpha, beta, data_ptr,
cols_ptr, block_ptr, num_rows, num_cols, left_size, right_size,
right_range_ptr,  x_ptr,y_ptr);
else
ell_multiply_kernel<value_type>  (alpha, beta, data_ptr, cols_ptr,
block_ptr, num_rows, num_cols, blocks_per_line, n, left_size,
right_size, right_range_ptr,  x_ptr,y_ptr);
}


template<class value_type>
void EllSparseBlockMatDevice<value_type>::launch_multiply_kernel( value_type alpha, const value_type* x_ptr, value_type beta, value_type* y_ptr) const
{
const value_type* data_ptr = thrust::raw_pointer_cast( &data[0]);
const int* cols_ptr = thrust::raw_pointer_cast( &cols_idx[0]);
const int* block_ptr = thrust::raw_pointer_cast( &data_idx[0]);
const int* right_range_ptr = thrust::raw_pointer_cast( &right_range[0]);
if( n == 1)
call_ell_multiply_kernel<value_type, 1>  (alpha, beta, data_ptr,
cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size,
right_size, right_range_ptr,  x_ptr,y_ptr);

else if( n == 2)
call_ell_multiply_kernel<value_type, 2>  (alpha, beta, data_ptr,
cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size,
right_size, right_range_ptr,  x_ptr,y_ptr);
else if( n == 3)
call_ell_multiply_kernel<value_type, 3>  (alpha, beta, data_ptr,
cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size,
right_size, right_range_ptr,  x_ptr,y_ptr);
else if( n == 4)
call_ell_multiply_kernel<value_type, 4>  (alpha, beta, data_ptr,
cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size,
right_size, right_range_ptr,  x_ptr,y_ptr);
else if( n == 5)
call_ell_multiply_kernel<value_type, 5>  (alpha, beta, data_ptr,
cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size,
right_size, right_range_ptr,  x_ptr,y_ptr);
else
ell_multiply_kernel<value_type> ( alpha, beta, data_ptr, cols_ptr,
block_ptr, num_rows, num_cols, blocks_per_line, n, left_size,
right_size, right_range_ptr,  x_ptr,y_ptr);
}

template<class value_type>
void coo_multiply_kernel( value_type alpha, const value_type** x, value_type beta, value_type* RESTRICT y, const CooSparseBlockMatDevice<value_type>& m )
{
#pragma omp for nowait
for (int skj = 0; skj < m.left_size*m.n*m.right_size; skj++)
{
int s = skj / (m.n*m.right_size);
int k = (skj % (m.n*m.right_size)) / m.right_size;
int j = (skj % (m.n*m.right_size)) % m.right_size;
for (int i = 0; i < m.num_entries; i++)
{
int I = ((s*m.num_rows + m.rows_idx[i])*m.n + k)*m.right_size + j;
value_type temp = 0;
for (int q = 0; q < m.n; q++) 
temp = DG_FMA(m.data[(m.data_idx[i] * m.n + k)*m.n + q],
x[m.cols_idx[i]][(q*m.left_size +s )*m.right_size+j],
temp);
y[I] = DG_FMA(alpha, temp, y[I]);
}
}
}
template<class value_type, int n>
void coo_multiply_kernel( value_type alpha, const value_type** x, value_type beta, value_type* RESTRICT y, const CooSparseBlockMatDevice<value_type>& m )
{
bool trivial = true;
int CC = m.cols_idx[0], DD = m.data_idx[0];
for( int i=0; i<m.num_entries; i++)
if( CC+i != m.cols_idx[i] || DD+i != m.data_idx[i])
trivial=false;
if( trivial)
{
#pragma omp for SIMD nowait
for (int sj = 0; sj < m.left_size*m.right_size; sj++)
{
int s = sj / m.right_size;
int j = (sj % m.right_size) % m.right_size;
for( int k=0; k<n; k++)
{
for (int i = 0; i < m.num_entries; i++)
{
int I = ((s*m.num_rows + m.rows_idx[i])*n + k)*m.right_size + j;
int DDD = ((DD +i)*n+k)*n, CCC = CC+i;
value_type temp = 0;
for (int q = 0; q < n; q++) 
temp = DG_FMA(m.data[DDD + q],
x[CCC][q*m.left_size*m.right_size +sj],
temp);
y[I] = DG_FMA(alpha, temp, y[I]);
}
}
}
}
else
{
#pragma omp for SIMD nowait
for (int sj = 0; sj < m.left_size*m.right_size; sj++)
{
int s = sj / m.right_size;
int j = (sj % m.right_size) % m.right_size;
for( int k=0; k<n; k++)
{
for (int i = 0; i < m.num_entries; i++)
{
int I = ((s*m.num_rows + m.rows_idx[i])*n + k)*m.right_size + j;
value_type temp = 0;
for (int q = 0; q < n; q++) 
temp = DG_FMA(m.data[(m.data_idx[i] * n + k)*n + q],
x[m.cols_idx[i]][q*m.left_size*m.right_size +sj],
temp);
y[I] = DG_FMA(alpha, temp, y[I]);
}
}
}
}
}
template<class value_type>
void CooSparseBlockMatDevice<value_type>::launch_multiply_kernel( value_type alpha, const value_type** x, value_type beta, value_type* RESTRICT y) const
{
if( n == 1)
coo_multiply_kernel<value_type, 1>( alpha, x, beta, y, *this);
else if( n == 2)
coo_multiply_kernel<value_type, 2>( alpha, x, beta, y, *this);
else if( n == 3)
coo_multiply_kernel<value_type, 3>( alpha, x, beta, y, *this);
else if( n == 4)
coo_multiply_kernel<value_type, 4>( alpha, x, beta, y, *this);
else
coo_multiply_kernel<value_type>( alpha, x, beta, y, *this);
}

}
