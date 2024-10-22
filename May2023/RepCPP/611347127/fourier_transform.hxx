#pragma once

#include <cstdio> 
#include <cmath> 
#include <cassert> 
#include <vector> 
#include <complex> 

#ifndef   HAS_NO_MKL
#include "mkl_dfti.h" 
#endif 

#ifdef    HAS_FFTW
extern "C" {
#include <fftw3.h> 
} 
#endif 

#ifndef   NO_UNIT_TESTS
#include "constants.hxx" 
#endif 

#include "status.hxx" 

namespace fourier_transform {

template <typename real_t>
status_t fft(real_t out[] 
, real_t out_imag[]
, real_t const in[] 
, real_t const in_imag[]
, int const ng[3] 
, bool const forward=true
, int const echo=0
) { 
#ifndef HAS_NO_MKL
MKL_LONG status;
MKL_LONG const l[3] = {ng[2], ng[1], ng[0]};
DFTI_DESCRIPTOR_HANDLE my_desc_handle;
status = DftiCreateDescriptor(&my_desc_handle, (sizeof(real_t) > 4) ? DFTI_DOUBLE : DFTI_SINGLE, DFTI_COMPLEX, 3, l);
status = DftiSetValue(my_desc_handle, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
status = DftiCommitDescriptor(my_desc_handle);
if (forward) { 
status = DftiComputeForward (my_desc_handle, (void*)in, (void*)in_imag, (void*)out, (void*)out_imag); 
} else {
status = DftiComputeBackward(my_desc_handle, (void*)in, (void*)in_imag, (void*)out, (void*)out_imag); 
}
DftiFreeDescriptor(&my_desc_handle); 
if (status != 0 && echo > 0) std::printf("# MKL-FFT returns status=%li\n", status);
return status;
#else 

#ifdef HAS_FFTW
size_t const ngall = size_t(ng[2]) * size_t(ng[1]) * size_t(ng[0]);
std::vector<std::complex<double>> cvi(ngall), cvo(ngall); 
for (size_t i = 0; i < ngall; ++i) { 
cvi[i] = std::complex<double>(in[i], in_imag[i]);
} 
auto const plan = fftw_plan_dft_3d(ng[2], ng[1], ng[0], (fftw_complex*) cvi.data(),
(fftw_complex*) cvo.data(),
forward ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
if (nullptr == plan) return __LINE__; 
fftw_execute(plan);
fftw_destroy_plan(plan);
for (size_t i = 0; i < ngall; ++i) { 
out[i]      = cvo[i].real();
out_imag[i] = cvo[i].imag();
} 
return 0; 
#endif 

return -1; 
#endif 
} 

inline status_t fft(std::complex<double> out[] 
, std::complex<double> const in[] 
, int const ng[3] 
, bool const forward=true
, int const echo=0) { 
#ifndef   HAS_NO_MKL
status_t status(-1);
if (echo > 0) std::printf("# MKL-FFT returns status=%i, not implemented\n", int(status));
return status;
#else  

#ifdef    HAS_FFTW
auto const plan = fftw_plan_dft_3d(ng[2], ng[1], ng[0], (fftw_complex*) in,
(fftw_complex*) out,
forward ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
if (nullptr == plan) return __LINE__; 
fftw_execute(plan);
fftw_destroy_plan(plan);
return 0; 
#endif 

return -1; 
#endif 
} 








#ifdef    NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else  

template <typename real_t>
inline status_t test_fft(int const echo=6) {
if (echo > 0) std::printf("\n# %s<%s>:\n", __func__, (8 == sizeof(real_t))?"double":"float");
int const ng[3] = {29, 13, 9};
int const ngall = ng[2]*ng[1]*ng[0];
std::vector<real_t> rs(2*ngall, real_t(0));
auto const rs_imag = rs.data() + ngall;
double const pw[3] = {3./ng[0], 2./ng[1], 1./ng[2]};
if (echo > 1) std::printf("# %s: set up a single plane wave as [%g %g %g]\n",
__func__, pw[0]*ng[0], pw[1]*ng[1], pw[2]*ng[2]);
for (int z = 0; z < ng[2]; ++z) {
for (int y = 0; y < ng[1]; ++y) {
for (int x = 0; x < ng[0]; ++x) {
int const i = (z*ng[1] + y)*ng[0] + x;
rs[i] = std::cos(2*constants::pi*((pw[0]*x + pw[1]*y + pw[2]*z)));
}}} 
std::vector<real_t> ft(2*ngall);
auto const ft_imag = ft.data() + ngall; 
auto const status_fft = fft(ft.data(), ft_imag, rs.data(), rs_imag, ng, true); 
real_t maximum = 0; int at[4] = {-1,-1,-1,-1};
for (int reim = 0; reim < 2; ++reim) {
for (int z = 0; z < ng[2]; ++z) {
for (int y = 0; y < ng[1]; ++y) {
for (int x = 0; x < ng[0]; ++x) {
int const i = (z*ng[1] + y)*ng[0] + x;
auto const fta = std::abs(ft[reim*ngall + i]);
if (fta > maximum) { maximum = fta; at[0] = x; at[1] = y; at[2] = z; at[3] = reim; }
}}}} 
if (echo > 5) std::printf("# %s: detected peak at index [%d %d %d] %s-part, value %g\n",
__func__, at[0], at[1], at[2], (at[3])?"imag":"real", maximum);
std::vector<real_t> rs_back(ngall);
auto const status_inv = fft(rs_back.data(), rs_imag, ft.data(), ft_imag, ng, false); 
if (echo > 8) std::printf("\n# %s: back-transformed cos-wave values:\n", __func__);
real_t const omega_inv = 1./ngall;
double deva = 0, dev2 = 0;
for (int z = 0; z < ng[2]; ++z) {
for (int y = 0; y < ng[1]; ++y) {
for (int x = 0; x < ng[0]; ++x) {
int const i = (z*ng[1] + y)*ng[0] + x;
auto const d = rs_back[i]*omega_inv - rs[i];
deva += std::abs(d); dev2 += d*d;
if (echo > 8) std::printf("%d %g %g %g\n", i, rs_back[i]*omega_inv, rs[i], d);
}}} 
if (echo > 2) std::printf("# back-transformed cos-wave differs abs %.1e rms %.1e\n", deva/ngall, std::sqrt(dev2/ngall));
if (echo > 1) std::printf("# %s: status = %i\n\n", __func__, int(status_fft) + int(status_inv));
return int(status_fft) + int(status_inv);
} 

inline status_t all_tests(int const echo=0) {
status_t status(0);
status += test_fft<float>(echo);
status += test_fft<double>(echo);
return status;
} 

#endif 

} 
