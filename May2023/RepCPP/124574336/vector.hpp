#ifndef  BOOST_SERIALIZATION_VECTOR_HPP
#define BOOST_SERIALIZATION_VECTOR_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <vector>

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/library_version_type.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/library_version_type.hpp>

#include <boost/serialization/collections_save_imp.hpp>
#include <boost/serialization/collections_load_imp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/array_wrapper.hpp>
#include <boost/mpl/bool_fwd.hpp>
#include <boost/mpl/if.hpp>

#ifndef BOOST_SERIALIZATION_VECTOR_VERSIONED
#define BOOST_SERIALIZATION_VECTOR_VERSIONED(V) (V==4 || V==5)
#endif

#if defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)
#define STD _STLP_STD
#else
#define STD std
#endif

namespace boost {
namespace serialization {



template<class Archive, class U, class Allocator>
inline void save(
Archive & ar,
const std::vector<U, Allocator> &t,
const unsigned int ,
mpl::false_
){
boost::serialization::stl::save_collection<Archive, STD::vector<U, Allocator> >(
ar, t
);
}

template<class Archive, class U, class Allocator>
inline void load(
Archive & ar,
std::vector<U, Allocator> &t,
const unsigned int ,
mpl::false_
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
t.reserve(count);
stl::collection_load_impl(ar, t, count, item_version);
}


template<class Archive, class U, class Allocator>
inline void save(
Archive & ar,
const std::vector<U, Allocator> &t,
const unsigned int ,
mpl::true_
){
const collection_size_type count(t.size());
ar << BOOST_SERIALIZATION_NVP(count);
if (!t.empty())
ar << serialization::make_array<const U, collection_size_type>(
static_cast<const U *>(&t[0]),
count
);
}

template<class Archive, class U, class Allocator>
inline void load(
Archive & ar,
std::vector<U, Allocator> &t,
const unsigned int ,
mpl::true_
){
collection_size_type count(t.size());
ar >> BOOST_SERIALIZATION_NVP(count);
t.resize(count);
unsigned int item_version=0;
if(BOOST_SERIALIZATION_VECTOR_VERSIONED(ar.get_library_version())) {
ar >> BOOST_SERIALIZATION_NVP(item_version);
}
if (!t.empty())
ar >> serialization::make_array<U, collection_size_type>(
static_cast<U *>(&t[0]),
count
);
}


template<class Archive, class U, class Allocator>
inline void save(
Archive & ar,
const std::vector<U, Allocator> &t,
const unsigned int file_version
){
typedef typename
boost::serialization::use_array_optimization<Archive>::template apply<
typename remove_const<U>::type
>::type use_optimized;
save(ar,t,file_version, use_optimized());
}

template<class Archive, class U, class Allocator>
inline void load(
Archive & ar,
std::vector<U, Allocator> &t,
const unsigned int file_version
){
#ifdef BOOST_SERIALIZATION_VECTOR_135_HPP
if (ar.get_library_version()==boost::serialization::library_version_type(5))
{
load(ar,t,file_version, boost::is_arithmetic<U>());
return;
}
#endif
typedef typename
boost::serialization::use_array_optimization<Archive>::template apply<
typename remove_const<U>::type
>::type use_optimized;
load(ar,t,file_version, use_optimized());
}

template<class Archive, class U, class Allocator>
inline void serialize(
Archive & ar,
std::vector<U, Allocator> & t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

template<class Archive, class Allocator>
inline void save(
Archive & ar,
const std::vector<bool, Allocator> &t,
const unsigned int 
){
collection_size_type count (t.size());
ar << BOOST_SERIALIZATION_NVP(count);
std::vector<bool>::const_iterator it = t.begin();
while(count-- > 0){
bool tb = *it++;
ar << boost::serialization::make_nvp("item", tb);
}
}

template<class Archive, class Allocator>
inline void load(
Archive & ar,
std::vector<bool, Allocator> &t,
const unsigned int 
){
collection_size_type count;
ar >> BOOST_SERIALIZATION_NVP(count);
t.resize(count);
for(collection_size_type i = collection_size_type(0); i < count; ++i){
bool b;
ar >> boost::serialization::make_nvp("item", b);
t[i] = b;
}
}

template<class Archive, class Allocator>
inline void serialize(
Archive & ar,
std::vector<bool, Allocator> & t,
const unsigned int file_version
){
boost::serialization::split_free(ar, t, file_version);
}

} 
} 

#include <boost/serialization/collection_traits.hpp>

BOOST_SERIALIZATION_COLLECTION_TRAITS(std::vector)
#undef STD

#endif 
