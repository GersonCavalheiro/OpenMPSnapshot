#ifndef BOOST_ARCHIVE_SERIALIZER_MAP_HPP
#define BOOST_ARCHIVE_SERIALIZER_MAP_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {

namespace serialization {
class extended_type_info;
} 

namespace archive {
namespace detail {

class basic_serializer;

template<class Archive>
class BOOST_SYMBOL_VISIBLE archive_serializer_map {
public:
static BOOST_ARCHIVE_OR_WARCHIVE_DECL bool insert(const basic_serializer * bs);
static BOOST_ARCHIVE_OR_WARCHIVE_DECL void erase(const basic_serializer * bs);
static BOOST_ARCHIVE_OR_WARCHIVE_DECL const basic_serializer * find(
const boost::serialization::extended_type_info & type_
);
};

} 
} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
