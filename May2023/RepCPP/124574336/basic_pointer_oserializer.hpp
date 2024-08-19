#ifndef BOOST_ARCHIVE_BASIC_POINTER_OSERIALIZER_HPP
#define BOOST_ARCHIVE_BASIC_POINTER_OSERIALIZER_HPP

#if defined(_MSC_VER)
# pragma once
#endif



#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/detail/basic_serializer.hpp>

#include <boost/archive/detail/abi_prefix.hpp> 

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace serialization {
class extended_type_info;
} 

namespace archive {
namespace detail {

class basic_oarchive;
class basic_oserializer;

class BOOST_SYMBOL_VISIBLE basic_pointer_oserializer :
public basic_serializer
{
protected:
explicit BOOST_ARCHIVE_DECL basic_pointer_oserializer(
const boost::serialization::extended_type_info & type_
);
public:
virtual BOOST_ARCHIVE_DECL ~basic_pointer_oserializer();
virtual const basic_oserializer & get_basic_serializer() const = 0;
virtual void save_object_ptr(
basic_oarchive & ar,
const void * x
) const = 0;
};

} 
} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
