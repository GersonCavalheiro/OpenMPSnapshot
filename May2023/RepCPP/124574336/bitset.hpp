
#ifndef BOOST_SERIALIZATION_BITSET_HPP
#define BOOST_SERIALIZATION_BITSET_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <bitset>
#include <cstddef> 

#include <boost/config.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/nvp.hpp>

namespace boost{
namespace serialization{

template <class Archive, std::size_t size>
inline void save(
Archive & ar,
std::bitset<size> const & t,
const unsigned int 
){
const std::string bits = t.template to_string<
std::string::value_type,
std::string::traits_type,
std::string::allocator_type
>();
ar << BOOST_SERIALIZATION_NVP( bits );
}

template <class Archive, std::size_t size>
inline void load(
Archive & ar,
std::bitset<size> & t,
const unsigned int 
){
std::string bits;
ar >> BOOST_SERIALIZATION_NVP( bits );
t = std::bitset<size>(bits);
}

template <class Archive, std::size_t size>
inline void serialize(
Archive & ar,
std::bitset<size> & t,
const unsigned int version
){
boost::serialization::split_free( ar, t, version );
}

template <std::size_t size>
struct tracking_level<std::bitset<size> >
: mpl::int_<track_never> {} ;

} 
} 

#endif 
