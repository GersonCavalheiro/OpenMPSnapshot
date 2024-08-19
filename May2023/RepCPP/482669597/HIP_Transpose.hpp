#ifndef __HIP_TRANSPOSE_HPP__
#define __HIP_TRANSPOSE_HPP__



#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas.h>
#include <type_traits>
#include "HIP_Helper.hpp"
#include "Layout.hpp"
#include "ComplexType.hpp"

template <typename RealType> using Complex = Impl::complex<RealType>;

namespace Impl {
template <typename ScalarType, class LayoutPolicy = layout_left,
std::enable_if_t<std::is_same_v<ScalarType, float           > ||
std::is_same_v<ScalarType, double          > ||
std::is_same_v<ScalarType, Complex<float>  > ||
std::is_same_v<ScalarType, Complex<double> > 
, std::nullptr_t> = nullptr>
struct Transpose {
private:
int col_;
int row_;
rocblas_handle handle_;

public:
using array_layout = LayoutPolicy;

public:
Transpose() = delete;

Transpose(int row, int col) : row_(row), col_(col) {
if(std::is_same_v<array_layout, layout_right>) {
row_ = col;
col_ = row;
}
SafeHIPCall( rocblas_create_handle(&handle_) );
SafeHIPCall( rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host) );
}

~Transpose() {
SafeHIPCall( rocblas_destroy_handle(handle_) );
}

void forward(ScalarType *dptr_in, ScalarType *dptr_out) {
#if defined(ENABLE_OPENMP_OFFLOAD)
#pragma omp target data use_device_ptr(dptr_in, dptr_out)
#endif
rocblasTranspose_(dptr_in, dptr_out, row_, col_);
SafeHIPCall( hipDeviceSynchronize() );
}

void backward(ScalarType *dptr_in, ScalarType *dptr_out) {
#if defined(ENABLE_OPENMP_OFFLOAD)
#pragma omp target data use_device_ptr(dptr_in, dptr_out)
#endif
rocblasTranspose_(dptr_in, dptr_out, col_, row_);
SafeHIPCall( hipDeviceSynchronize() );
}

template <typename SType=ScalarType,
std::enable_if_t<std::is_same_v<SType, float>, std::nullptr_t> = nullptr>
void rocblasTranspose_(SType *dptr_in, SType *dptr_out, int row, int col) {
constexpr float alpha = 1.0;
constexpr float beta  = 0.0;
SafeHIPCall( 
rocblas_sgeam(handle_,                     
rocblas_operation_transpose, 
rocblas_operation_transpose, 
col,                         
row,                         
&alpha,                      
dptr_in,                     
row,                         
&beta,                       
dptr_in,                     
row,                         
dptr_out,                    
col)                         
);
}

template <typename SType=ScalarType, 
std::enable_if_t<std::is_same_v<SType, double>, std::nullptr_t> = nullptr>
void rocblasTranspose_(SType *dptr_in, SType *dptr_out, int row, int col) {
constexpr double alpha = 1.0;
constexpr double beta  = 0.0;
SafeHIPCall( 
rocblas_dgeam(handle_,                     
rocblas_operation_transpose, 
rocblas_operation_transpose, 
col,                         
row,                         
&alpha,                      
dptr_in,                     
row,                         
&beta,                       
dptr_in,                     
row,                         
dptr_out,                    
col)                         
);
}

template <typename SType=ScalarType,
std::enable_if_t<std::is_same_v<SType, Complex<float> >, std::nullptr_t> = nullptr>
void rocblasTranspose_(SType *dptr_in, SType *dptr_out, int row, int col) {
#if defined(ENABLE_OPENMP_OFFLOAD)
rocblas_float_complex alpha; alpha.x = 1.0;
rocblas_float_complex beta;
#else
const rocblas_float_complex alpha(1.0);
const rocblas_float_complex beta(0.0);
#endif
SafeHIPCall( 
rocblas_cgeam(handle_,                     
rocblas_operation_transpose, 
rocblas_operation_transpose, 
col,                         
row,                         
&alpha,                      
reinterpret_cast<rocblas_float_complex*>(dptr_in), 
row,                         
&beta,                       
reinterpret_cast<rocblas_float_complex*>(dptr_in), 
row,                         
reinterpret_cast<rocblas_float_complex*>(dptr_out), 
col)                         
);
}

template <typename SType=ScalarType, 
std::enable_if_t<std::is_same_v<SType, Complex<double> >, std::nullptr_t> = nullptr>
void rocblasTranspose_(SType *dptr_in, SType *dptr_out, int row, int col) {
#if defined(ENABLE_OPENMP_OFFLOAD)
rocblas_double_complex alpha; alpha.x = 1.0;
rocblas_double_complex beta;
#else
const rocblas_double_complex alpha(1.0);
const rocblas_double_complex beta(0.0);
#endif
SafeHIPCall( 
rocblas_zgeam(handle_,                     
rocblas_operation_transpose, 
rocblas_operation_transpose, 
col,                         
row,                         
&alpha,                      
reinterpret_cast<rocblas_double_complex*>(dptr_in), 
row,                         
&beta,                       
reinterpret_cast<rocblas_double_complex*>(dptr_in), 
row,                         
reinterpret_cast<rocblas_double_complex*>(dptr_out), 
col)                         
);
}
};
};

#endif
