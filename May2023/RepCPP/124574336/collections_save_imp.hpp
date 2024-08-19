#ifndef BOOST_SERIALIZATION_COLLECTIONS_SAVE_IMP_HPP
#define BOOST_SERIALIZATION_COLLECTIONS_SAVE_IMP_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>
#include <boost/core/addressof.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/item_version_type.hpp>

namespace boost{
namespace serialization {
namespace stl {


template<class Archive, class Container>
inline void save_collection(
Archive & ar,
const Container &s,
collection_size_type count)
{
ar << BOOST_SERIALIZATION_NVP(count);
const item_version_type item_version(
version<typename Container::value_type>::value
);

ar << BOOST_SERIALIZATION_NVP(item_version);

typename Container::const_iterator it = s.begin();
while(count-- > 0){
boost::serialization::save_construct_data_adl(
ar,
boost::addressof(*it),
item_version
);
ar << boost::serialization::make_nvp("item", *it++);
}
}

template<class Archive, class Container>
inline void save_collection(Archive & ar, const Container &s)
{
collection_size_type count(s.size());
save_collection(ar, s, count);
}

} 
} 
} 

#endif 
