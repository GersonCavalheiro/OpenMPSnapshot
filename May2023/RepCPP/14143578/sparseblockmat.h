#pragma once

#include <cmath>
#include <thrust/host_vector.h>
#include <cusp/coo_matrix.h>
#include "exblas/exdot_serial.h"
#include "config.h"
#include "exceptions.h"
#include "tensor_traits.h"
#include "tensor_traits.h"

namespace dg
{


template<class value_type>
struct EllSparseBlockMat
{
EllSparseBlockMat() = default;

EllSparseBlockMat( int num_block_rows, int num_block_cols,
int num_blocks_per_line, int num_different_blocks, int n):
data(num_different_blocks*n*n),
cols_idx( num_block_rows*num_blocks_per_line),
data_idx(cols_idx.size()), right_range(2),
num_rows(num_block_rows),
num_cols(num_block_cols),
blocks_per_line(num_blocks_per_line),
n(n), left_size(1), right_size(1)
{
right_range[0]=0;
right_range[1]=1;
}
int total_num_rows()const{
return num_rows*n*left_size*right_size;
}
int total_num_cols()const{
return num_cols*n*left_size*right_size;
}


cusp::coo_matrix<int, value_type, cusp::host_memory> asCuspMatrix() const;


void symv(SharedVectorTag, SerialTag, value_type alpha, const value_type* RESTRICT x, value_type beta, value_type* RESTRICT y) const;

void set_default_range(){
right_range[0]=0;
right_range[1]=right_size;
}
void set_right_size( int new_right_size ){
right_size = new_right_size;
set_default_range();
}
void set_left_size( int new_left_size ){
left_size = new_left_size;
}

void display( std::ostream& os = std::cout, bool show_data = false) const;

thrust::host_vector<value_type> data;
thrust::host_vector<int> cols_idx; 
thrust::host_vector<int> data_idx; 
thrust::host_vector<int> right_range; 
int num_rows; 
int num_cols; 
int blocks_per_line; 
int n;  
int left_size; 
int right_size; 

};



template<class value_type>
struct CooSparseBlockMat
{
CooSparseBlockMat() = default;

CooSparseBlockMat( int num_block_rows, int num_block_cols, int n, int left_size, int right_size):
num_rows(num_block_rows), num_cols(num_block_cols), num_entries(0),
n(n),left_size(left_size), right_size(right_size){}


void add_value( int row, int col, const thrust::host_vector<value_type>& element)
{
assert( (int)element.size() == n*n);
int index = data.size()/n/n;
data.insert( data.end(), element.begin(), element.end());
rows_idx.push_back(row);
cols_idx.push_back(col);
data_idx.push_back( index );

num_entries++;
}

int total_num_rows()const{
return num_rows*n*left_size*right_size;
}
int total_num_cols()const{
return num_cols*n*left_size*right_size;
}




void symv(SharedVectorTag, SerialTag, value_type alpha, const value_type** x, value_type beta, value_type* RESTRICT y) const;

void display(std::ostream& os = std::cout, bool show_data = false) const;

thrust::host_vector<value_type> data;
thrust::host_vector<int> cols_idx; 
thrust::host_vector<int> rows_idx; 
thrust::host_vector<int> data_idx; 
int num_rows; 
int num_cols; 
int num_entries; 
int n;  
int left_size; 
int right_size; 
};

template<class value_type>
void EllSparseBlockMat<value_type>::symv(SharedVectorTag, SerialTag, value_type alpha, const value_type* RESTRICT x, value_type beta, value_type* RESTRICT y) const
{
for( int s=0; s<left_size; s++)
for( int i=0; i<num_rows; i++)
for( int k=0; k<n; k++)
for( int j=right_range[0]; j<right_range[1]; j++)
{
int I = ((s*num_rows + i)*n+k)*right_size+j;
y[I] = beta == 0 ? (value_type)0 : y[I]*beta;
for( int d=0; d<blocks_per_line; d++)
{
value_type temp = 0;
for( int q=0; q<n; q++) 
temp = DG_FMA( data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q],
x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)*right_size+j],
temp);
y[I] = DG_FMA( alpha,temp, y[I]);
}
}
}
template<class value_type>
cusp::coo_matrix<int, value_type, cusp::host_memory> EllSparseBlockMat<value_type>::asCuspMatrix() const
{
cusp::array1d<value_type, cusp::host_memory> values;
cusp::array1d<int, cusp::host_memory> row_indices;
cusp::array1d<int, cusp::host_memory> column_indices;
for( int s=0; s<left_size; s++)
for( int i=0; i<num_rows; i++)
for( int k=0; k<n; k++)
for( int j=right_range[0]; j<right_range[1]; j++)
{
int I = ((s*num_rows + i)*n+k)*right_size+j;
for( int d=0; d<blocks_per_line; d++)
for( int q=0; q<n; q++) 
{
row_indices.push_back(I);
column_indices.push_back(
((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)*right_size+j);
values.push_back(data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]);
}
}
cusp::coo_matrix<int, value_type, cusp::host_memory> A(
total_num_rows(), total_num_cols(), values.size());
A.row_indices = row_indices;
A.column_indices = column_indices;
A.values = values;
return A;
}

template<class value_type>
void CooSparseBlockMat<value_type>::symv( SharedVectorTag, SerialTag, value_type alpha, const value_type** x, value_type beta, value_type* RESTRICT y) const
{
if( num_entries==0)
return;
if( beta!= 1 )
std::cerr << "Beta != 1 yields wrong results in CooSparseBlockMat!! Beta = "<<beta<<"\n";
assert( beta == 1 && "Beta != 1 yields wrong results in CooSparseBlockMat!!");

for( int s=0; s<left_size; s++)
for( int k=0; k<n; k++)
for( int j=0; j<right_size; j++)
for( int i=0; i<num_entries; i++)
{
value_type temp = 0;
for( int q=0; q<n; q++) 
temp = DG_FMA( data[ (data_idx[i]*n + k)*n+q],
x[cols_idx[i]][(q*left_size +s )*right_size+j],
temp);
int I = ((s*num_rows + rows_idx[i])*n+k)*right_size+j;
y[I] = DG_FMA( alpha,temp, y[I]);
}
}

template<class T>
void EllSparseBlockMat<T>::display( std::ostream& os, bool show_data ) const
{
os << "Data array has   "<<data.size()/n/n<<" blocks of size "<<n<<"x"<<n<<"\n";
os << "num_rows         "<<num_rows<<"\n";
os << "num_cols         "<<num_cols<<"\n";
os << "blocks_per_line  "<<blocks_per_line<<"\n";
os << "n                "<<n<<"\n";
os << "left_size             "<<left_size<<"\n";
os << "right_size            "<<right_size<<"\n";
os << "right_range_0         "<<right_range[0]<<"\n";
os << "right_range_1         "<<right_range[1]<<"\n";
os << "Column indices: \n";
for( int i=0; i<num_rows; i++)
{
for( int d=0; d<blocks_per_line; d++)
os << cols_idx[i*blocks_per_line + d] <<" ";
os << "\n";
}
os << "\n Data indices: \n";
for( int i=0; i<num_rows; i++)
{
for( int d=0; d<blocks_per_line; d++)
os << data_idx[i*blocks_per_line + d] <<" ";
os << "\n";
}
if(show_data)
{
os << "\n Data: \n";
for( unsigned i=0; i<data.size()/n/n; i++)
for(unsigned k=0; k<n*n; k++)
{
dg::exblas::udouble res;
res.d = data[i*n*n+k];
os << "idx "<<i<<" "<<res.d <<"\t"<<res.i<<"\n";
}
}
os << std::endl;
}

template<class value_type>
void CooSparseBlockMat<value_type>::display( std::ostream& os, bool show_data) const
{
os << "Data array has   "<<data.size()/n/n<<" blocks of size "<<n<<"x"<<n<<"\n";
os << "num_rows         "<<num_rows<<"\n";
os << "num_cols         "<<num_cols<<"\n";
os << "num_entries      "<<num_entries<<"\n";
os << "n                "<<n<<"\n";
os << "left_size             "<<left_size<<"\n";
os << "right_size            "<<right_size<<"\n";
os << "row\tcolumn\tdata:\n";
for( int i=0; i<num_entries; i++)
os << rows_idx[i]<<"\t"<<cols_idx[i] <<"\t"<<data_idx[i]<<"\n";
if(show_data)
{
os << "\n Data: \n";
for( unsigned i=0; i<data.size()/n/n; i++)
for(unsigned k=0; k<n*n; k++)
{
dg::exblas::udouble res;
res.d = data[i*n*n+k];
os << "idx "<<i<<" "<<res.d <<"\t"<<res.i<<"\n";
}
}
os << std::endl;

}

template <class T>
struct TensorTraits<EllSparseBlockMat<T> >
{
using value_type  = T;
using tensor_category = SparseBlockMatrixTag;
};
template <class T>
struct TensorTraits<CooSparseBlockMat<T> >
{
using value_type  = T;
using tensor_category = SparseBlockMatrixTag;
};

} 
