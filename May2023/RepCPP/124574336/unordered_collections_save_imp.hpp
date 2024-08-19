#ifndef BOOST_SERIALIZATION_UNORDERED_COLLECTIONS_SAVE_IMP_HPP
#define BOOST_SERIALIZATION_UNORDERED_COLLECTIONS_SAVE_IMP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif





#include <boost/config.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/library_version_type.hpp>

namespace boost{
namespace serialization {
namespace stl {


template<class Archive, class Container>
inline void save_unordered_collection(Archive & ar, const Container &s)
{
collection_size_type count(s.size());
const collection_size_type bucket_count(s.bucket_count());
const item_version_type item_version(
version<typename Container::value_type>::value
);

#if 0

boost::serialization::library_version_type library_version(
ar.get_library_version()
);
ar << BOOST_SERIALIZATION_NVP(count);
ar << BOOST_SERIALIZATION_NVP(bucket_count);
if(boost::serialization::library_version_type(3) < library_version){
ar << BOOST_SERIALIZATION_NVP(item_version);
}
#else
ar << BOOST_SERIALIZATION_NVP(count);
ar << BOOST_SERIALIZATION_NVP(bucket_count);
ar << BOOST_SERIALIZATION_NVP(item_version);
#endif

typename Container::const_iterator it = s.begin();
while(count-- > 0){
boost::serialization::save_construct_data_adl(
ar,
&(*it),
boost::serialization::version<
typename Container::value_type
>::value
);
ar << boost::serialization::make_nvp("item", *it++);
}
}

} 
} 
} 

#endif 
