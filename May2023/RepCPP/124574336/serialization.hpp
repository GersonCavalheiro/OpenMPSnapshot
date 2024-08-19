#ifndef BOOST_SERIALIZATION_SERIALIZATION_HPP
#define BOOST_SERIALIZATION_SERIALIZATION_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#if defined(_MSC_VER)
#  pragma warning (disable : 4675) 
#endif

#include <boost/config.hpp>
#include <boost/serialization/strong_typedef.hpp>







#include <boost/serialization/access.hpp>


namespace boost {
namespace serialization {

BOOST_STRONG_TYPEDEF(unsigned int, version_type)

template<class Archive, class T>
inline void serialize(
Archive & ar, T & t, const unsigned int file_version
){
access::serialize(ar, t, static_cast<unsigned int>(file_version));
}

template<class Archive, class T>
inline void save_construct_data(
Archive & ,
const T * ,
const unsigned int 
){
}

template<class Archive, class T>
inline void load_construct_data(
Archive & ,
T * t,
const unsigned int 
){
access::construct(t);
}


template<class Archive, class T>
inline void serialize_adl(
Archive & ar,
T & t,
const unsigned int file_version
){
const version_type v(file_version);
serialize(ar, t, v);
}

template<class Archive, class T>
inline void save_construct_data_adl(
Archive & ar,
const T * t,
const unsigned int file_version
){

const version_type v(file_version);
save_construct_data(ar, t, v);
}

template<class Archive, class T>
inline void load_construct_data_adl(
Archive & ar,
T * t,
const unsigned int file_version
){
const version_type v(file_version);
load_construct_data(ar, t, v);
}

} 
} 

#endif 
