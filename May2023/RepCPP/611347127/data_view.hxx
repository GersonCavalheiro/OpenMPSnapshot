#pragma once

#include <cstdio> 
#include <cassert> 
#include <cstdint> 
#include <algorithm> 
#include <utility> 

#include "status.hxx" 
#include "complex_tools.hxx" 
#include "recorded_warnings.hxx" 

#define debug_printf(...)


#ifdef DEVEL
#ifndef NO_UNIT_TESTS
#include "simple_timer.hxx" 
#endif 
#endif 

#define DimUnknown 0

namespace data_view {
inline void _check_index(int const srcline, size_t const n, size_t const i, char const d) {
if (i >= n) error("# data_view.hxx:%d i%c=%ld >= n%c=%ld\n", srcline, '0'+d, i, '0'+d, n);
assert(i < n);
} 
} 

#define CHECK_INDEX(n,i,d) data_view::_check_index(__LINE__, n, i, d);

template <typename T>
class view2D {
public:

view2D() : _data(nullptr), _n0(DimUnknown), _n1(DimUnknown), _mem(0) { 
} 

view2D(T* const ptr, size_t const stride) 
: _data(ptr), _n0(stride), _n1(DimUnknown), _mem(0) { 
} 

view2D(size_t const n1, size_t const stride, T const init_value={0}) 
: _data(new T[n1*stride]), _n0(stride), _n1(n1), _mem(n1*stride*sizeof(T)) {
debug_printf("# view2D(n1=%i, stride=%i [, init_value]) constructor allocates %g kByte\n", n1, stride, _mem*.001);
std::fill(_data, _data + n1*stride, init_value); 
} 

~view2D() {
if (_data && (_mem > 0)) {
debug_printf("# ~view2D() destructor tries to free %g kByte\n", _mem*.001);
delete[] _data;
} 
} 

view2D(view2D<T> && rhs) {
debug_printf("# view2D(view2D<T> && rhs);\n");
*this = std::move(rhs);
} 

view2D(view2D<T> const & rhs) = delete;

view2D& operator= (view2D<T> && rhs) {
debug_printf("# view2D& operator= (view2D<T> && rhs);\n");
_data = rhs._data;
_n0   = rhs._n0;
_n1   = rhs._n1;
_mem  = rhs._mem; rhs._mem = 0; 
return *this;
} 

view2D& operator= (view2D<T> const & rhs) = delete;


#define _VIEW2D_HAS_PARENTHESIS
#ifdef  _VIEW2D_HAS_PARENTHESIS
T const & operator () (size_t const i1, size_t const i0) const {
if (_n1 > DimUnknown)
CHECK_INDEX(_n1, i1, 1);
CHECK_INDEX(_n0, i0, 0);
return _data[i1*_n0 + i0]; } 

T       & operator () (size_t const i1, size_t const i0)       {
if (_n1 > DimUnknown)
CHECK_INDEX(_n1, i1, 1);
CHECK_INDEX(_n0, i0, 0);
return _data[i1*_n0 + i0]; } 
#endif 

T* operator[] (size_t const i1) const {
if (_n1 > DimUnknown)
CHECK_INDEX(_n1, i1, 1);
return &_data[i1*_n0]; } 

T* data() const { return _data; }
size_t stride() const { return _n0; }

private:
T * _data;
size_t _n0, _n1; 
size_t _mem; 

}; 


template <typename T>
inline void set(view2D<T> & y, size_t const n1, T const a) { 
std::fill(y.data(), y.data() + n1*y.stride(), a);
} 


template <typename T>
view2D<T> transpose(
view2D<T> const & a 
, int const aN 
, int const aM=-1 
, char const conj='n' 
) {

int const N = (-1 == aM) ? a.stride() : aM;
int const M = aN;
assert( N <= a.stride() );
bool const c = ('c' == (conj | 32)); 
view2D<T> a_transposed(N, M);
for (int n = 0; n < N; ++n) {
for (int m = 0; m < M; ++m) {
auto const a_mn = a(m,n); 
a_transposed(n,m) = c ? conjugate(a_mn) : a_mn;
} 
} 
return a_transposed;
} 

template <typename Ta, typename Tb, typename Tc>
void gemm(view2D<Tc> & c 
, int const N, view2D<Tb> const & b 
, int const K, view2D<Ta> const & a 
, int const aM=-1 
, char const beta='0' 
) {

int const M = (-1 == aM) ? std::min(c.stride(), a.stride()) : aM;
if (M > a.stride()) error("M= %d > %ld =a.stride", M, a.stride());
if (K > b.stride()) error("K= %d > %ld =b.stride", M, b.stride());
if (M > c.stride()) error("M= %d > %ld =c.stride", M, c.stride());
assert( M <= a.stride() );
assert( K <= b.stride() );
assert( M <= c.stride() );
for (int n = 0; n < N; ++n) {
for (int m = 0; m < M; ++m) {
Tc t(0);
for (int k = 0; k < K; ++k) { 
t += b(n,k) * a(k,m);
} 
if ('0' == beta) { c(n,m) = t; } else { c(n,m) += t; } 
} 
} 
} 



template <typename T>
class view3D {
public:

view3D() : _data(nullptr), _n0(0), _n1(0), _n2(DimUnknown), _mem(0) { } 

view3D(T* const ptr, size_t const n1, size_t const stride)
: _data(ptr), _n0(stride), _n1(n1), _n2(DimUnknown), _mem(0) { } 

view3D(size_t const n2, size_t const n1, size_t const stride, T const init_value={0}) 
: _data(new T[n2*n1*stride]), _n0(stride), _n1(n1), _n2(n2), _mem(n2*n1*stride*sizeof(T)) {
debug_printf("# view3D(n2=%i, n1=%i, stride=%i [, init_value]) constructor allocates %g kByte\n", n2, n1, stride, _mem*.001);
std::fill(_data, _data + n2*n1*stride, init_value); 
} 

~view3D() { 
if (_data && (_mem > 0)) {
delete[] _data;
debug_printf("# ~view3D() destructor tries to free %g kByte\n", _mem*.001);
}
} 

view3D(view3D<T>      && rhs) { 
debug_printf("# view3D(view3D<T> && rhs);\n");
*this = std::move(rhs);
} 

view3D(view3D<T> const & rhs) = delete; 

view3D& operator= (view3D<T> && rhs) {
debug_printf("# view3D& operator= (view3D<T> && rhs);\n");
_data = rhs._data;
_n0   = rhs._n0;
_n1   = rhs._n1;
_n2   = rhs._n2;
_mem  = rhs._mem; rhs._mem = 0; 
return *this;
} 

view3D& operator= (view3D<T> const & rhs) = delete;

#define _VIEW3D_HAS_PARENTHESIS
#ifdef  _VIEW3D_HAS_PARENTHESIS
#define _access return _data[(i2*_n1 + i1)*_n0 + i0]
T const & operator () (size_t const i2, size_t const i1, size_t const i0) const {
if (_n2 > DimUnknown)
CHECK_INDEX(_n2, i2, 2);
CHECK_INDEX(_n1, i1, 1);
CHECK_INDEX(_n0, i0, 0);
_access; }

T       & operator () (size_t const i2, size_t const i1, size_t const i0)       {
if (_n2 > DimUnknown)
CHECK_INDEX(_n2, i2, 2);
CHECK_INDEX(_n1, i1, 1);
CHECK_INDEX(_n0, i0, 0);
_access; }
#undef _access
#endif 

#define _VIEW3D_HAS_PARENTHESIS_2ARGS
#ifdef  _VIEW3D_HAS_PARENTHESIS_2ARGS
#define _access return &_data[(i2*_n1 + i1)*_n0]
T* operator () (size_t const i2, size_t const i1) const {
if (_n2 > DimUnknown)
CHECK_INDEX(_n2, i2, 2);
CHECK_INDEX(_n1, i1, 1);
_access; }
#undef _access
#endif 

#define _VIEW3D_HAS_INDEXING
#ifdef  _VIEW3D_HAS_INDEXING
view2D<T> operator[] (size_t const i2) const { 
if (_n2 > DimUnknown)
CHECK_INDEX(_n2, i2, 2);
return view2D<T>(_data + i2*_n1*_n0, _n0); } 
#endif 

T* data() const { return _data; }
size_t stride() const { return _n0; }
size_t dim1()   const { return _n1; }
bool is_memory_owner() const { return (_n2 > DimUnknown); }

private:
T* _data;
size_t _n0, _n1, _n2; 
size_t _mem; 

}; 

template <typename T>
inline void set(view3D<T> & y, size_t const n2, T const a) { 
std::fill(y.data(), y.data() + n2*y.dim1()*y.stride(), a);
} 


template <typename T>
class view4D {
public:

view4D() : _data(nullptr), _n0(0), _n1(0), _n2(0), _n3(DimUnknown) { } 

view4D(T* const ptr, size_t const n2, size_t const n1, size_t const stride) 
: _data(ptr), _n0(stride), _n1(n1), _n2(n2), _n3(DimUnknown) { } 

view4D(size_t const n3, size_t const n2, size_t const n1, size_t const stride, T const init_value={0}) 
: _data(new T[n3*n2*n1*stride]), _n0(stride), _n1(n1), _n2(n2), _n3(n3), _mem(n3*n2*n1*stride*sizeof(T)) {
debug_printf("# view4D(n3=%i, n2=%i, n1=%i, stride=%i [, init_value]) constructor allocates %g kByte\n", n3, n2, n1, stride, _mem*.001);
std::fill(_data, _data + n3*n2*n1*stride, init_value); 
} 

~view4D() { 
if (_data && (_mem > 0)) {
delete[] _data;
debug_printf("# ~view4D() destructor tries to free %g kByte\n", _mem*.001);
}
} 

view4D(view4D<T> && rhs) { 
debug_printf("# view4D(view4D<T> && rhs);\n");
*this = std::move(rhs);
} 

view4D(view4D<T> const & rhs) = delete;

view4D& operator= (view4D<T> && rhs) {
debug_printf("# view4D& operator= (view4D<T> && rhs);\n");
_data = rhs._data;
_n0   = rhs._n0;
_n1   = rhs._n1;
_n2   = rhs._n2;
_n3   = rhs._n3;
_mem  = rhs._mem; rhs._mem = 0; 
return *this;
} 

view4D& operator= (view4D<T> const & rhs) = delete;

#define _VIEW4D_HAS_PARENTHESIS
#ifdef  _VIEW4D_HAS_PARENTHESIS
#define _access return _data[((i3*_n2 + i2)*_n1 + i1)*_n0 + i0]
T const & operator () (size_t const i3, size_t const i2, size_t const i1, size_t const i0) const {
if (_n3 > DimUnknown)
CHECK_INDEX(_n3, i3, 3);
CHECK_INDEX(_n2, i2, 2);
CHECK_INDEX(_n1, i1, 1);
CHECK_INDEX(_n0, i0, 0);
_access; }

T       & operator () (size_t const i3, size_t const i2, size_t const i1, size_t const i0)       {
if (_n3 > DimUnknown)
CHECK_INDEX(_n3, i3, 3);
CHECK_INDEX(_n2, i2, 2);
CHECK_INDEX(_n1, i1, 1);
CHECK_INDEX(_n0, i0, 0);
_access; }
#undef _access
#endif 

#define _VIEW4D_HAS_PARENTHESIS_3ARGS
#ifdef  _VIEW4D_HAS_PARENTHESIS_3ARGS
#define _access return &_data[((i3*_n2 + i2)*_n1 + i1)*_n0]
T* operator () (size_t const i3, size_t const i2, size_t const i1) const {
if (_n3 > DimUnknown)
CHECK_INDEX(_n3, i3, 3);
CHECK_INDEX(_n2, i2, 2);
CHECK_INDEX(_n1, i1, 1);
_access; }
#undef _access
#endif 

#define _VIEW4D_HAS_PARENTHESIS_2ARGS
#ifdef  _VIEW4D_HAS_PARENTHESIS_2ARGS
view2D<T> operator () (size_t const i3, size_t const i2) const {
if (_n3 > DimUnknown)
CHECK_INDEX(_n3, i3, 3);
CHECK_INDEX(_n2, i2, 2);
return view2D<T>(_data + (i3*_n2 + i2)*_n1*_n0, _n0); }
#endif 

#define _VIEW4D_HAS_INDEXING
#ifdef  _VIEW4D_HAS_INDEXING
view3D<T> operator[] (size_t const i3) const {
if (_n3 > DimUnknown)
CHECK_INDEX(_n3, i3, 3);
return view3D<T>(_data + i3*_n2*_n1*_n0, _n1, _n0); } 
#endif 

T* data() const { return _data; }
size_t stride() const { return _n0; }
size_t dim1()   const { return _n1; }
size_t dim2()   const { return _n2; }
bool is_memory_owner() const { return (_n3 > DimUnknown); }

private:
T* _data;
size_t _n0, _n1, _n2, _n3; 
size_t _mem; 

}; 

template <typename T>
inline void set(view4D<T> & y, size_t const n3, T const a) { 
std::fill(y.data(), y.data() + n3*y.dim2()*y.dim1()*y.stride(), a); }

#undef DimUnknown
#undef debug_printf
#undef CHECK_INDEX

namespace data_view {

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline int test_view2D(int const echo=9) {
int constexpr n1 = 3, n0 = 5;
if (echo > 3) std::printf("\n# %s(%i,%i)\n", __func__, n1, n0);

auto a = view2D<double>(n1,8);
assert(a.stride() >= n0);
for (int i = 0; i < n1; ++i) {
for (int j = 0; j < n0; ++j) {
#ifdef _VIEW2D_HAS_PARENTHESIS
a(i,j) = i + 0.1*j;
if (echo > 3) std::printf("# a2D(%i,%i) = %g\n", i,j, a(i,j));
assert(a(i,j) == a[i][j]);
#else
#error "view2D needs parenthesis!"
#endif
} 
} 

int ii = 1;
if (echo > 3) std::printf("\n# ai = a1D[%i][:]\n", ii);
auto const ai = a[ii]; 
for (int j = 0; j < n0; ++j) {
if (echo > 3) std::printf("# ai[%i] = %g\n", j, ai[j]);
#ifdef _VIEW2D_HAS_PARENTHESIS
assert(a(ii,j) == ai[j]);
#else
#error "view2D needs parenthesis!"
#endif
} 

return 0;
} 

inline int test_view3D(int const echo=9) {
int constexpr n2 = 3, n1 = 2, n0 = 5;
if (echo > 3) std::printf("\n# %s(%i,%i,%i)\n", __func__, n2, n1, n0);
view3D<float> a(n2,n1,8); 
assert(a.stride() >= n0);
for (int h = 0; h < n2; ++h) {
for (int i = 0; i < n1; ++i) {
for (int j = 0; j < n0; ++j) {
a(h,i,j) = h + 0.1*i + 0.01*j;
if (echo > 5) std::printf("# a3D(%i,%i,%i) = %g\n", h,i,j, a(h,i,j));
#ifdef _VIEW3D_HAS_INDEXING
assert(a(h,i,j) == a[h][i][j]);
#endif
} 
} 
} 

return 0;
} 

inline int test_view4D(int const echo=9) {
int constexpr n3 = 1, n2 = 2, n1 = 3, n0 = 4;
if (echo > 3) std::printf("\n# %s(%i,%i,%i,%i)\n", __func__, n3, n2, n1, n0);
view4D<float> a(n3,n2,n1,n0); 
assert(a.stride() >= n0);
for (int g = 0; g < n3; ++g) {
for (int h = 0; h < n2; ++h) {
for (int i = 0; i < n1; ++i) {
for (int j = 0; j < n0; ++j) {
a(g,h,i,j) = g*10 + h + 0.1*i + 0.01*j;
if (echo > 7) std::printf("# a4D(%i,%i,%i,%i) = %g\n", g,h,i,j, a(g,h,i,j));
#ifdef _VIEW4D_HAS_INDEXING
assert(a(g,h,i,j) == a[g][h][i][j]);
#endif
} 
} 
} 
} 

return 0;
} 

inline status_t test_bench_view2D(int const echo=1, int const nrep=1e7) {
#ifdef DEVEL
if (echo < 1) return 0;
view2D<int> a(2, 2, 0);
{   SimpleTimer t(__FILE__, __LINE__, "a[i][j]", echo);
for (int irep = 0; irep < nrep; ++irep) {
a[1][1] = a[1][0];
a[1][0] = a[0][1];
a[0][1] = a[0][0];
a[0][0] = irep;
} 
} 
if (echo > 3) std::printf("# a[i][j] = %i %i %i %i\n", a(0,0), a(0,1), a(1,0), a(1,1));
std::fflush(stdout);
{   SimpleTimer t(__FILE__, __LINE__, "a(i,j)", echo);
for (int irep = 0; irep < nrep; ++irep) {
a(1,1) = a(1,0);
a(1,0) = a(0,1); 
a(0,1) = a(0,0);
a(0,0) = irep;
} 
} 
if (echo > 3) std::printf("# a(i,j)  = %i %i %i %i\n", a(0,0), a(0,1), a(1,0), a(1,1));
std::fflush(stdout);
#endif 
return 0;
} 

inline status_t all_tests(int const echo=0) {
status_t status = 0;
status += test_view2D(echo);
status += test_view3D(echo);
status += test_view4D(echo);
status += test_bench_view2D(echo);
return status;
} 

#endif 

} 

