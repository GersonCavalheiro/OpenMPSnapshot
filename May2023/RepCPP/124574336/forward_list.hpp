#ifndef BOOST_SERIALIZATION_FORWARD_LIST_HPP
#define BOOST_SERIALIZATION_FORWARD_LIST_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>

#include <forward_list>
#include <iterator>  

#include <boost/serialization/collections_save_imp.hpp>
#include <boost/serialization/collections_load_imp.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/library_version_type.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/detail/stack_constructor.hpp>
#include <boost/serialization/detail/is_default_constructible.hpp>
#include <boost/move/utility_core.hpp>

namespace boost {
namespace serialization {

template<class Archive, class U, class Allocator>
inline void save(
Archive & ar,
const std::forward_list<U, Allocator> &t,
const unsigned int 
){
const collection_size_type count(std::distance(t.cbegin(), t.cend()));
boost::serialization::stl::save_collection<
Archive,
std::forward_list<U, Allocator>
>(ar, t, count);
}

namespace stl {

template<
class Archive,
class T,
class Allocator
>
typename boost::disable_if<
typename detail::is_default_constructible<
typename std::forward_list<T, Allocator>::value_type
>,
void
>::type
collection_load_impl(
Archive & ar,
std::forward_list<T, Allocator> &t,
collection_size_type count,
item_version_type item_version
){
t.clear();
boost::serialization::detail::stack_construct<Archive, T> u(ar, item_version);
ar >> boost::serialization::make_nvp("item", u.reference());
t.push_front(boost::move(u.reference()));
typename std::forward_list<T, Allocator>::iterator last;
last = t.begin();
ar.reset_object_address(&(*t.begin()) , & u.reference());
while(--count > 0){
detail::stack_construct<Archive, T> u(ar, item_version);
ar >> boost::serialization::make_nvp("item", u.reference());
last = t.insert_after(last, boost::move(u.reference()));
ar.reset_object_address(&(*last) , & u.reference());
}
}

} 

template<class Archive, class U, class Allocator>
inline void load(
Archive & ar,
std::forward_list<U, Allocator> &t,
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
std::forward_list<U, Allocator> &t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 

#include <boost/serialization/collection_traits.hpp>

BOOST_SERIALIZATION_COLLECTION_TRAITS(std::forward_list)

#endif  
