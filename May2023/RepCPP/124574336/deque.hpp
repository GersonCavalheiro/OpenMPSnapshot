#ifndef  BOOST_SERIALIZATION_DEQUE_HPP
#define BOOST_SERIALIZATION_DEQUE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <deque>

#include <boost/config.hpp>

#include <boost/serialization/library_version_type.hpp>
#include <boost/serialization/collections_save_imp.hpp>
#include <boost/serialization/collections_load_imp.hpp>
#include <boost/serialization/split_free.hpp>

namespace boost {
namespace serialization {

template<class Archive, class U, class Allocator>
inline void save(
Archive & ar,
const std::deque<U, Allocator> &t,
const unsigned int 
){
boost::serialization::stl::save_collection<
Archive, std::deque<U, Allocator>
>(ar, t);
}

template<class Archive, class U, class Allocator>
inline void load(
Archive & ar,
std::deque<U, Allocator> &t,
const unsigned int 
){
const boost::serialization::library_version_type library_version(
ar.get_library_version()
);
item_version_type item_version(0);
collection_size_type count;
ar >> BOOST_SERIALIZATION_NVP(count);
if(boost::serialization::library_version_type(3) < library_version){
ar >> BOOST_SERIALIZATION_NVP(item_version);
}
stl::collection_load_impl(ar, t, count, item_version);
}

template<class Archive, class U, class Allocator>
inline void serialize(
Archive & ar,
std::deque<U, Allocator> &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 

#include <boost/serialization/collection_traits.hpp>

BOOST_SERIALIZATION_COLLECTION_TRAITS(std::deque)

#endif 
