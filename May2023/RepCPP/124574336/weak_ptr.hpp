#ifndef BOOST_SERIALIZATION_WEAK_PTR_HPP
#define BOOST_SERIALIZATION_WEAK_PTR_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/weak_ptr.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace boost {
namespace serialization{

template<class Archive, class T>
inline void save(
Archive & ar,
const boost::weak_ptr< T > &t,
const unsigned int 
){
const boost::shared_ptr< T > sp = t.lock();
ar << boost::serialization::make_nvp("weak_ptr", sp);
}

template<class Archive, class T>
inline void load(
Archive & ar,
boost::weak_ptr< T > &t,
const unsigned int 
){
boost::shared_ptr< T > sp;
ar >> boost::serialization::make_nvp("weak_ptr", sp);
t = sp;
}

template<class Archive, class T>
inline void serialize(
Archive & ar,
boost::weak_ptr< T > &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 

#ifndef BOOST_NO_CXX11_SMART_PTR
#include <memory>

namespace boost {
namespace serialization{

template<class Archive, class T>
inline void save(
Archive & ar,
const std::weak_ptr< T > &t,
const unsigned int 
){
const std::shared_ptr< T > sp = t.lock();
ar << boost::serialization::make_nvp("weak_ptr", sp);
}

template<class Archive, class T>
inline void load(
Archive & ar,
std::weak_ptr< T > &t,
const unsigned int 
){
std::shared_ptr< T > sp;
ar >> boost::serialization::make_nvp("weak_ptr", sp);
t = sp;
}

template<class Archive, class T>
inline void serialize(
Archive & ar,
std::weak_ptr< T > &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 

#endif 

#endif 
