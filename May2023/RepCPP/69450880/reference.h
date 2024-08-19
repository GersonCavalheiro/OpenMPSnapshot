

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/use_default.h>
#include <hydra/detail/external/hydra_thrust/detail/reference_forward_declaration.h>
#include <ostream>


namespace hydra_thrust
{
namespace detail
{

template<typename> struct is_wrapped_reference;

}

template<typename Element, typename Pointer, typename Derived>
class reference
{
private:
typedef typename hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_same<Derived,use_default>::value,
hydra_thrust::detail::identity_<reference>,
hydra_thrust::detail::identity_<Derived>
>::type derived_type;

struct wrapped_reference_hint {};
template<typename> friend struct hydra_thrust::detail::is_wrapped_reference;

public:
typedef Pointer                                              pointer;
typedef typename hydra_thrust::detail::remove_const<Element>::type value_type;

__host__ __device__
explicit reference(const pointer &ptr);

template<typename OtherElement, typename OtherPointer, typename OtherDerived>
__host__ __device__
reference(const reference<OtherElement,OtherPointer,OtherDerived> &other,
typename hydra_thrust::detail::enable_if_convertible<
typename reference<OtherElement,OtherPointer,OtherDerived>::pointer,
pointer
>::type * = 0);

__host__ __device__
derived_type &operator=(const reference &other);

template<typename OtherElement, typename OtherPointer, typename OtherDerived>
__host__ __device__
derived_type &operator=(const reference<OtherElement,OtherPointer,OtherDerived> &other);

__host__ __device__
derived_type &operator=(const value_type &x);

__host__ __device__
pointer operator&() const;

__host__ __device__
operator value_type () const;

__host__ __device__
void swap(derived_type &other);

derived_type &operator++();

value_type operator++(int);

derived_type &operator+=(const value_type &rhs);

derived_type &operator--();

value_type operator--(int);

derived_type &operator-=(const value_type &rhs);

derived_type &operator*=(const value_type &rhs);

derived_type &operator/=(const value_type &rhs);

derived_type &operator%=(const value_type &rhs);

derived_type &operator<<=(const value_type &rhs);

derived_type &operator>>=(const value_type &rhs);

derived_type &operator&=(const value_type &rhs);

derived_type &operator|=(const value_type &rhs);

derived_type &operator^=(const value_type &rhs);

private:
const pointer m_ptr;

template <typename OtherElement, typename OtherPointer, typename OtherDerived> friend class reference;

template<typename System>
__host__ __device__
inline value_type strip_const_get_value(const System &system) const;

template<typename OtherPointer>
__host__ __device__
inline void assign_from(OtherPointer src);

template<typename System1, typename System2, typename OtherPointer>
inline __host__ __device__
void assign_from(System1 *system1, System2 *system2, OtherPointer src);

template<typename System, typename OtherPointer>
__host__ __device__
inline void strip_const_assign_value(const System &system, OtherPointer src);

template<typename System>
inline __host__ __device__
void swap(System *system, derived_type &other);

template<typename System>
inline __host__ __device__
value_type convert_to_value_type(System *system) const;
}; 

template<typename Element, typename Pointer, typename Derived,
typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
const reference<Element, Pointer, Derived> &y);

} 

#include <hydra/detail/external/hydra_thrust/detail/reference.inl>

