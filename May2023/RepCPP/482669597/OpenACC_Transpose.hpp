#ifndef __OPENACC_TRANSPOSE_HPP__
#define __OPENACC_TRANSPOSE_HPP__

#include <cublas_v2.h>
#include <complex>
#include <experimental/mdspan>

template <typename RealType> using Complex = std::complex<RealType>;
namespace stdex = std::experimental;

namespace Impl {
template <typename RealType, class LayoutPolicy = stdex::layout_left,
std::enable_if_t<std::is_same_v<RealType, float           > ||
std::is_same_v<RealType, double          > ||
std::is_same_v<RealType, Complex<float>  > ||
std::is_same_v<RealType, Complex<double> >
, std::nullptr_t> = nullptr
>
struct Transpose {
private:
int col_;
int row_;
cublasHandle_t handle_;

public:
using array_layout = LayoutPolicy;

public:
Transpose() = delete;

Transpose(int row, int col) : row_(row), col_(col) {
if(std::is_same_v<array_layout, stdex::layout_right>) {
row_ = col;
col_ = row;
}
cublasCreate(&handle_);
}

~Transpose() {
cublasDestroy(handle_);
}

void forward(RealType *dptr_in, RealType *dptr_out) {
#pragma acc host_data use_device(dptr_in, dptr_out)
cublasTranspose_(dptr_in, dptr_out, row_, col_);
}

void backward(RealType *dptr_in, RealType *dptr_out) {
#pragma acc host_data use_device(dptr_in, dptr_out)
cublasTranspose_(dptr_in, dptr_out, col_, row_);
}

private:
template <typename RType=RealType,
std::enable_if_t<std::is_same_v<RType, float>, std::nullptr_t> = nullptr>
void cublasTranspose_(RType *dptr_in, RType *dptr_out, int row, int col) {
constexpr float alpha = 1.0;
constexpr float beta  = 0.0;
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

template <typename RType=RealType,
std::enable_if_t<std::is_same_v<RType, double>, std::nullptr_t> = nullptr>
void cublasTranspose_(RType *dptr_in, RType *dptr_out, int row, int col) {
constexpr double alpha = 1.;
constexpr double beta  = 0.;
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

template <typename RType=RealType,
std::enable_if_t<std::is_same_v<RType, Complex<float> >, std::nullptr_t> = nullptr>
void cublasTranspose_(RType *dptr_in, RType *dptr_out, int row, int col) {
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

template <typename RType=RealType,
std::enable_if_t<std::is_same_v<RType, Complex<double> >, std::nullptr_t> = nullptr>
void cublasTranspose_(RType *dptr_in, RType *dptr_out, int row, int col) {
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
