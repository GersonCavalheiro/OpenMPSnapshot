#ifndef __HIP_FFT_HPP__
#define __HIP_FFT_HPP__



#include <vector>
#include <rocfft.h>
#include <type_traits>
#include "HIP_Helper.hpp"
#include "Layout.hpp"
#include "ComplexType.hpp"

template <typename RealType> using Complex = Impl::complex<RealType>;

namespace Impl {
template <typename RealType, class LayoutPolicy = layout_left,
typename std::enable_if<std::is_same<RealType, float>::value ||
std::is_same<RealType, double>::value 
>::type * = nullptr> 
struct FFT {
private:
rocfft_plan forward_plan_, backward_plan_;

rocfft_execution_info forward_execution_info_, backward_execution_info_;

rocfft_plan_description forward_description_, backward_description_;

void *forward_wbuffer_, *backward_wbuffer_;

rocfft_status rc_;

int nx1_;

int nx2_;

int nb_batches_;

int nx1h_;

int nx2h_;

RealType normcoeff_;

public:
using array_layout = LayoutPolicy;

public:
FFT(int nx1, int nx2)
: nx1_(nx1), nx2_(nx2), nb_batches_(1), rc_(rocfft_status_success) {
init();
}

FFT(int nx1, int nx2, int nb_batches)
: nx1_(nx1), nx2_(nx2), nb_batches_(nb_batches), rc_(rocfft_status_success) {
init();
}

virtual ~FFT() {
rocfft_execution_info_destroy(forward_execution_info_);
rocfft_execution_info_destroy(backward_execution_info_);
rocfft_plan_description_destroy(forward_description_);
rocfft_plan_description_destroy(backward_description_);
rocfft_plan_destroy(forward_plan_);
rocfft_plan_destroy(backward_plan_);

if(forward_wbuffer_ != nullptr) SafeHIPCall( hipFree(forward_wbuffer_) );
if(backward_wbuffer_ != nullptr) SafeHIPCall( hipFree(backward_wbuffer_) );
}

RealType normcoeff() const {return normcoeff_;}
int nx1() {return nx1_;}
int nx2() {return nx2_;}
int nx1h() {return nx1h_;}
int nx2h() {return nx2h_;}
int nb_batches() {return nb_batches_;}

void rfft2(RealType *dptr_in, Complex<RealType> *dptr_out) {
#if defined(ENABLE_OPENMP_OFFLOAD)
#pragma omp target data use_device_ptr(dptr_in, dptr_out)
#endif
rc_ = rocfft_execute(forward_plan_,            
(void**)&dptr_in,         
(void**)&dptr_out,        
forward_execution_info_); 
if(rc_ != rocfft_status_success)
throw std::runtime_error("failed to execute");
}

void irfft2(Complex<RealType> *dptr_in, RealType *dptr_out) {
#if defined(ENABLE_OPENMP_OFFLOAD)
#pragma omp target data use_device_ptr(dptr_in, dptr_out)
#endif
rc_ = rocfft_execute(backward_plan_,            
(void**)&dptr_in,          
(void**)&dptr_out,         
backward_execution_info_); 
if(rc_ != rocfft_status_success)
throw std::runtime_error("failed to execute");
}

private:
void init() {
static_assert(std::is_same<array_layout, layout_left>::value, "The input Layout must be LayoutLeft");
nx1h_ = nx1_/2 + 1;
nx2h_ = nx2_/2 + 1;

std::vector<size_t> length = {static_cast<size_t>(nx1_), static_cast<size_t>(nx2_)};
normcoeff_ = static_cast<RealType>(1.0) / static_cast<RealType>(nx1_ * nx2_);

const rocfft_result_placement place = rocfft_placement_notinplace;

std::vector<size_t> rstride = {1};

for(int i = 1; i < length.size(); i++) {
auto val = length[i-1] * rstride[i-1];
rstride.push_back(val);
}

std::vector<size_t> clength = length;
clength[0]                  = clength[0] / 2 + 1;
std::vector<size_t> cstride = {1};

for(int i = 1; i < clength.size(); ++i) {
cstride.push_back(clength[i - 1] * cstride[i - 1]);
}

const std::vector<size_t> ilength = length;
const std::vector<size_t> istride = rstride;

const std::vector<size_t> olength = clength;
const std::vector<size_t> ostride = cstride;

rc_ = rocfft_plan_description_create(&forward_description_);
rc_ = rocfft_plan_description_create(&backward_description_);

if (rc_ != rocfft_status_success)
throw std::runtime_error("device error");

rc_ = rocfft_plan_description_set_data_layout(
forward_description_,
rocfft_array_type_real, 
rocfft_array_type_hermitian_interleaved, 
nullptr,
nullptr,
istride.size(), 
istride.data(), 
0,              
ostride.size(), 
ostride.data(), 
0);             

rc_ = rocfft_plan_description_set_data_layout(
backward_description_,
rocfft_array_type_hermitian_interleaved, 
rocfft_array_type_real, 
nullptr,
nullptr,
ostride.size(), 
ostride.data(), 
0,              
istride.size(), 
istride.data(), 
0);             

if(rc_ != rocfft_status_success)
throw std::runtime_error("failed to set data layout");

rocfft_precision precision;
if(std::is_same<RealType, float>::value) {
precision = rocfft_precision_single;
}

if(std::is_same<RealType, double>::value) {
precision = rocfft_precision_double;
}

rc_ = rocfft_plan_create(&forward_plan_,
place,
rocfft_transform_type_real_forward,
precision,
length.size(), 
length.data(), 
nb_batches_, 
forward_description_); 

rc_ = rocfft_plan_create(&backward_plan_,
place,
rocfft_transform_type_real_inverse,
precision,
length.size(), 
length.data(), 
nb_batches_, 
backward_description_); 

if(rc_ != rocfft_status_success)
throw std::runtime_error("failed to create plan");

set_execution_info(&forward_wbuffer_, forward_plan_, forward_execution_info_);
set_execution_info(&backward_wbuffer_, backward_plan_, backward_execution_info_);
}

void set_execution_info(void **wbuffer, rocfft_plan &plan, rocfft_execution_info &execution_info) {
rc_ = rocfft_execution_info_create(&execution_info);
if(rc_ != rocfft_status_success)
throw std::runtime_error("failed to create execution info");

size_t workbuffersize = 0;
rc_ = rocfft_plan_get_work_buffer_size(plan, &workbuffersize);
if(rc_ != rocfft_status_success)
throw std::runtime_error("failed to get work buffer size");

*wbuffer = nullptr;
if(workbuffersize > 0) {
auto hip_status = hipMalloc(&wbuffer, workbuffersize);
if(hip_status != hipSuccess)
throw std::runtime_error("hipMalloc failed");

rc_ = rocfft_execution_info_set_work_buffer(execution_info, wbuffer, workbuffersize);
if(rc_ != rocfft_status_success)
throw std::runtime_error("failed to set work buffer");
}
}
};
};

#endif
