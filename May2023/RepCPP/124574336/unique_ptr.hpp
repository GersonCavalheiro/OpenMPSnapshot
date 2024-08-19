#ifndef  BOOST_SERIALIZATION_UNIQUE_PTR_HPP
#define BOOST_SERIALIZATION_UNIQUE_PTR_HPP

#if defined(_MSC_VER)
# pragma once
#endif



#include <memory>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/nvp.hpp>

namespace boost {
namespace serialization {

template<class Archive, class T>
inline void save(
Archive & ar,
const std::unique_ptr< T > &t,
const unsigned int 
){
const T * const tx = t.get();
ar << BOOST_SERIALIZATION_NVP(tx);
}

template<class Archive, class T>
inline void load(
Archive & ar,
std::unique_ptr< T > &t,
const unsigned int 
){
T *tx;
ar >> BOOST_SERIALIZATION_NVP(tx);
t.reset(tx);
}

template<class Archive, class T>
inline void serialize(
Archive & ar,
std::unique_ptr< T > &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 


#endif 
