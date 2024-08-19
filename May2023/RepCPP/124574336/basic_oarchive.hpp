#ifndef BOOST_ARCHIVE_BASIC_OARCHIVE_HPP
#define BOOST_ARCHIVE_BASIC_OARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <cstddef> 
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

#include <boost/archive/basic_archive.hpp>
#include <boost/serialization/tracking_enum.hpp>
#include <boost/archive/detail/helper_collection.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace serialization {
class extended_type_info;
} 

namespace archive {
namespace detail {

class basic_oarchive_impl;
class basic_oserializer;
class basic_pointer_oserializer;

class BOOST_SYMBOL_VISIBLE basic_oarchive :
private boost::noncopyable,
public boost::archive::detail::helper_collection
{
friend class basic_oarchive_impl;
boost::scoped_ptr<basic_oarchive_impl> pimpl;

virtual void vsave(const version_type t) =  0;
virtual void vsave(const object_id_type t) =  0;
virtual void vsave(const object_reference_type t) =  0;
virtual void vsave(const class_id_type t) =  0;
virtual void vsave(const class_id_optional_type t) = 0;
virtual void vsave(const class_id_reference_type t) =  0;
virtual void vsave(const class_name_type & t) = 0;
virtual void vsave(const tracking_type t) = 0;
protected:
BOOST_ARCHIVE_DECL basic_oarchive(unsigned int flags = 0);
BOOST_ARCHIVE_DECL boost::archive::detail::helper_collection &
get_helper_collection();
virtual BOOST_ARCHIVE_DECL ~basic_oarchive();
public:
BOOST_ARCHIVE_DECL void register_basic_serializer(
const basic_oserializer & bos
);
BOOST_ARCHIVE_DECL void save_object(
const void *x,
const basic_oserializer & bos
);
BOOST_ARCHIVE_DECL void save_pointer(
const void * t,
const basic_pointer_oserializer * bpos_ptr
);
void save_null_pointer(){
vsave(BOOST_SERIALIZATION_NULL_POINTER_TAG);
}
BOOST_ARCHIVE_DECL void end_preamble(); 
BOOST_ARCHIVE_DECL boost::serialization::library_version_type get_library_version() const;
BOOST_ARCHIVE_DECL unsigned int get_flags() const;
};

} 
} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
