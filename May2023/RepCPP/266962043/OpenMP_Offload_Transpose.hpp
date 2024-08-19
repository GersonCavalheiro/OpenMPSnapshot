#ifndef __OPENMP_OFFLOAD_TRANSPOSE_HPP__
#define __OPENMP_OFFLOAD_TRANSPOSE_HPP__

#include <cublas_v2.h>
#include <complex>
#include <iostream>
#include "ViewLayout.hpp"

template <typename RealType> using Complex = std::complex<RealType>;

namespace Impl {
template <typename ScalarType, Layout LayoutType = Layout::LayoutLeft,
typename std::enable_if<std::is_same<ScalarType, float           >::value ||
std::is_same<ScalarType, double          >::value ||
std::is_same<ScalarType, Complex<float>  >::value ||
std::is_same<ScalarType, Complex<double> >::value
>::type * = nullptr>

struct Transpose {
private:
int col_;
int row_;
cublasHandle_t handle_;

public:
typedef std::integral_constant<Layout, LayoutType> layout_;

public:

Transpose(int row, int col) {
typedef std::integral_constant<Layout, Layout::LayoutLeft> layout_left;
if(std::is_same<layout_, layout_left>::value) {
row_ = row;
col_ = col;
} else {
row_ = col;
col_ = row;
}

cublasCreate(&handle_);
}

~Transpose() {
cublasDestroy(handle_);
}


void sync_() {
int dummy[1];
#pragma omp target map(to: dummy)
{}
}

void forward(ScalarType *dptr_in, ScalarType *dptr_out) {
#pragma omp target data use_device_ptr(dptr_in, dptr_out)
cublasTranspose_(dptr_in, dptr_out, row_, col_);
sync_();
}

void backward(ScalarType *dptr_in, ScalarType *dptr_out) {
#pragma omp target data use_device_ptr(dptr_in, dptr_out)
cublasTranspose_(dptr_in, dptr_out, col_, row_);
sync_();
}

private:
void cublasTranspose_(float *dptr_in, float *dptr_out, int row, int col) {
const float alpha = 1.0;
const float beta  = 0.0;
cublasSgeam(handle_,     
CUBLAS_OP_T, 
CUBLAS_OP_T, 
col,         
row,         
&alpha,      
dptr_in,     
row,         
&beta,       
dptr_in,     
row,         
dptr_out,    
col);        
}

void cublasTranspose_(double *dptr_in, double *dptr_out, int row, int col) {
const double alpha = 1.;
const double beta  = 0.;
cublasDgeam(handle_,     
CUBLAS_OP_T, 
CUBLAS_OP_T, 
col,         
row,         
&alpha,      
dptr_in,     
row,         
&beta,       
dptr_in,     
row,         
dptr_out,    
col);        
}

void cublasTranspose_(Complex<float> *dptr_in, Complex<float> *dptr_out, int row, int col) {
const cuComplex alpha = make_cuComplex(1.0, 0.0);
const cuComplex beta  = make_cuComplex(0.0, 0.0);
cublasCgeam(handle_,     
CUBLAS_OP_T, 
CUBLAS_OP_N, 
col,         
row,         
&alpha,      
reinterpret_cast<cuComplex*>(dptr_in), 
row,         
&beta,       
reinterpret_cast<cuComplex*>(dptr_in), 
row,         
reinterpret_cast<cuComplex*>(dptr_out), 
col);        
}

void cublasTranspose_(Complex<double> *dptr_in, Complex<double> *dptr_out, int row, int col) {
const cuDoubleComplex alpha = make_cuDoubleComplex(1., 0.);
const cuDoubleComplex beta  = make_cuDoubleComplex(0., 0.);
cublasZgeam(handle_,     
CUBLAS_OP_T, 
CUBLAS_OP_N, 
col,         
row,         
&alpha,      
reinterpret_cast<cuDoubleComplex*>(dptr_in),  
row,         
&beta,       
reinterpret_cast<cuDoubleComplex*>(dptr_in),  
row,         
reinterpret_cast<cuDoubleComplex*>(dptr_out), 
col);        
}
};
};

#endif
