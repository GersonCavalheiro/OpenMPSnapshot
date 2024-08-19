#ifndef  BOOST_SERIALIZATION_COMPLEX_HPP
#define BOOST_SERIALIZATION_COMPLEX_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <complex>
#include <boost/config.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/serialization/split_free.hpp>

namespace boost {
namespace serialization {

template<class Archive, class T>
inline void serialize(
Archive & ar,
std::complex< T > & t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

template<class Archive, class T>
inline void save(
Archive & ar,
std::complex< T > const & t,
const unsigned int 
){
const T re = t.real();
const T im = t.imag();
ar << boost::serialization::make_nvp("real", re);
ar << boost::serialization::make_nvp("imag", im);
}

template<class Archive, class T>
inline void load(
Archive & ar,
std::complex< T >& t,
const unsigned int 
){
T re;
T im;
ar >> boost::serialization::make_nvp("real", re);
ar >> boost::serialization::make_nvp("imag", im);
t = std::complex< T >(re,im);
}

template <class T>
struct is_bitwise_serializable<std::complex< T > >
: public is_bitwise_serializable< T > {};

template <class T>
struct implementation_level<std::complex< T > >
: mpl::int_<object_serializable> {} ;

template <class T>
struct tracking_level<std::complex< T > >
: mpl::int_<track_never> {} ;

} 
} 

#endif 
