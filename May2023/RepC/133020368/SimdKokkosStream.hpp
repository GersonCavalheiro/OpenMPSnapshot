#pragma once
#include <iostream>
#include <stdexcept>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include "Stream.h"
template <class T>
class SimdKokkosStream : public Stream<T>
{
public:
unsigned int array_size_scalar;
unsigned int array_size_vector;
using simd_t      = typename Kokkos::Experimental::native_simd<T>;
using simd_mask_t = typename Kokkos::Experimental::native_simd_mask<T>;
using view_t = Kokkos::View<T*>;
using mirror_view_t = typename view_t::HostMirror;
protected:
view_t d_a;
view_t d_b;
view_t d_c;
mirror_view_t hm_a;
mirror_view_t hm_b;
mirror_view_t hm_c;
public:
SimdKokkosStream(const unsigned int, const int);
~SimdKokkosStream();
virtual void copy() override;
virtual void add() override;
virtual void mul() override;
virtual void triad() override;
virtual T dot() override;
virtual void init_arrays(T initA, T initB, T initC) override;
virtual void read_arrays(
std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
}; 
namespace Kokkos { 
template<>
struct reduction_identity< SimdKokkosStream<double>::simd_t > {
KOKKOS_FORCEINLINE_FUNCTION static SimdKokkosStream<double>::simd_t sum() {
return SimdKokkosStream<double>::simd_t(1.0);
}
};
} 
template <class T, class Space>
struct SimdReducer {
public:
using simd_t = typename SimdKokkosStream<T>::simd_t;
using reducer = SimdReducer<T, Space>;
using value_type = simd_t;
using result_view_type = Kokkos::View<value_type, Space, Kokkos::MemoryUnmanaged>;
private:
result_view_type value;
public:
KOKKOS_INLINE_FUNCTION
SimdReducer(value_type& value_) : value(&value_) {}
KOKKOS_INLINE_FUNCTION
void join(value_type& dest, const value_type& src) const {
dest = dest + src;
}
KOKKOS_INLINE_FUNCTION
void join(volatile value_type& dest, const volatile value_type& src) const {
dest += src;
}
KOKKOS_INLINE_FUNCTION
void init(value_type& val) const {
val = simd_t(0.0);
}
KOKKOS_INLINE_FUNCTION
value_type& reference() const { return *value.data(); }
KOKKOS_INLINE_FUNCTION
result_view_type view() const { return value; }
KOKKOS_INLINE_FUNCTION
bool references_scalar() const { return true; }
};
template<class T, class simd_type>
KOKKOS_INLINE_FUNCTION
bool
load(const T* ptr, std::size_t n, simd_type& simd)
{
if (n < simd.size()) return false;
simd.copy_from( ptr, Kokkos::Experimental::element_aligned_tag() );
return true;
}
template<class T, class simd_type>
KOKKOS_INLINE_FUNCTION
bool
store( T* ptr, std::size_t n, const simd_type& simd )
{
if (n < simd.size()) return false;
simd.copy_to( ptr, Kokkos::Experimental::element_aligned_tag() );
return true;
}
template <class T>
SimdKokkosStream<T>::SimdKokkosStream(
const unsigned int ARRAY_SIZE, const int device_index)
: array_size_scalar(ARRAY_SIZE),
array_size_vector(ARRAY_SIZE/simd_t::size()),
d_a("d_a", array_size_scalar),
d_b("d_b", array_size_scalar),
d_c("d_c", array_size_scalar),
hm_a(create_mirror_view(d_a)),
hm_b(create_mirror_view(d_b)),
hm_c(create_mirror_view(d_c))
{
printf("SimdKokkosStream: simd_t::size = %ld\n",simd_t::size());
}
template <class T>
SimdKokkosStream<T>::~SimdKokkosStream()
{
}
template <class T>
void SimdKokkosStream<T>::init_arrays(T initA, T initB, T initC)
{
Kokkos::parallel_for(array_size_vector, KOKKOS_CLASS_LAMBDA (const long index)
{
const auto sindex = index*simd_t::size();
store( &d_a(sindex) , simd_t::size(), simd_t(initA));
store( &d_b(sindex) , simd_t::size(), simd_t(initB));
store( &d_c(sindex) , simd_t::size(), simd_t(initC));
});
Kokkos::fence();
}
template <class T>
void SimdKokkosStream<T>::read_arrays(
std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
deep_copy(hm_a, d_a);
deep_copy(hm_b, d_b);
deep_copy(hm_c, d_c);
for(int ii = 0; ii < array_size_scalar; ++ii)
{
a[ii] = (hm_a)(ii);
b[ii] = (hm_b)(ii);
c[ii] = (hm_c)(ii);
}
}
template <class T>
void SimdKokkosStream<T>::copy()
{
Kokkos::parallel_for(array_size_vector, KOKKOS_CLASS_LAMBDA (const long index)
{
const auto sindex = index*simd_t::size();
simd_t value;
load(&d_a(sindex), simd_t::size(), value);
store(&d_c(sindex), simd_t::size(), value);
});
Kokkos::fence();
}
template <class T>
void SimdKokkosStream<T>::mul()
{
const T scalar = startScalar;
Kokkos::parallel_for(array_size_vector, KOKKOS_CLASS_LAMBDA (const long index)
{
const auto sindex = index*simd_t::size();
simd_t value;
load(&d_c(sindex), simd_t::size(), value);
value *= scalar;
store(&d_b(sindex), simd_t::size(), value);
});
Kokkos::fence();
}
template <class T>
void SimdKokkosStream<T>::add()
{
Kokkos::parallel_for(array_size_vector, KOKKOS_CLASS_LAMBDA (const long index)
{
const auto sindex = index*simd_t::size();
simd_t value1, value2;
load(&d_a(sindex), simd_t::size(), value1);
load(&d_b(sindex), simd_t::size(), value2);
const auto value = value1 + value2;
store( &d_c(sindex), simd_t::size(), value);
});
Kokkos::fence();
}
template <class T>
void SimdKokkosStream<T>::triad()
{
const T scalar = startScalar;
Kokkos::parallel_for(array_size_vector, KOKKOS_CLASS_LAMBDA (const long index)
{
const auto sindex = index*simd_t::size();
simd_t value1, value2;
load(&d_b(sindex), simd_t::size(), value1);
load(&d_c(sindex), simd_t::size(), value2);
const auto value = value1 + scalar*value2;
store( &d_a(sindex), simd_t::size(), value);
});
Kokkos::fence();
}
template <class T>
T SimdKokkosStream<T>::dot()
{
return 0;
}
