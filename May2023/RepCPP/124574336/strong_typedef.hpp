#ifndef BOOST_SERIALIZATION_STRONG_TYPEDEF_HPP
#define BOOST_SERIALIZATION_STRONG_TYPEDEF_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>
#include <boost/operators.hpp>
#include <boost/type_traits/has_nothrow_assign.hpp>
#include <boost/type_traits/has_nothrow_constructor.hpp>
#include <boost/type_traits/has_nothrow_copy.hpp>

#define BOOST_STRONG_TYPEDEF(T, D)                                                                               \
struct D                                                                                                         \
: boost::totally_ordered1< D                                                                                 \
, boost::totally_ordered2< D, T                                                                              \
> >                                                                                                          \
{                                                                                                                \
T t;                                                                                                         \
explicit D(const T& t_) BOOST_NOEXCEPT_IF(boost::has_nothrow_copy_constructor<T>::value) : t(t_) {}          \
D() BOOST_NOEXCEPT_IF(boost::has_nothrow_default_constructor<T>::value) : t() {}                             \
D(const D & t_) BOOST_NOEXCEPT_IF(boost::has_nothrow_copy_constructor<T>::value) : t(t_.t) {}                \
D& operator=(const D& rhs) BOOST_NOEXCEPT_IF(boost::has_nothrow_assign<T>::value) {t = rhs.t; return *this;} \
D& operator=(const T& rhs) BOOST_NOEXCEPT_IF(boost::has_nothrow_assign<T>::value) {t = rhs; return *this;}   \
operator const T&() const {return t;}                                                                        \
operator T&() {return t;}                                                                                    \
bool operator==(const D& rhs) const {return t == rhs.t;}                                                     \
bool operator<(const D& rhs) const {return t < rhs.t;}                                                       \
};

#endif 
