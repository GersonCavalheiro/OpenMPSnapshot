#ifndef BOOST_ARCHIVE_DETAIL_BASIC_IARCHIVE_HPP
#define BOOST_ARCHIVE_DETAIL_BASIC_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

#include <boost/serialization/tracking_enum.hpp>
#include <boost/archive/basic_archive.hpp>
#include <boost/archive/detail/decl.hpp>
#include <boost/archive/detail/helper_collection.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace serialization {
class extended_type_info;
} 

namespace archive {
namespace detail {

class basic_iarchive_impl;
class basic_iserializer;
class basic_pointer_iserializer;

class BOOST_SYMBOL_VISIBLE basic_iarchive :
private boost::noncopyable,
public boost::archive::detail::helper_collection
{
friend class basic_iarchive_impl;
boost::scoped_ptr<basic_iarchive_impl> pimpl;

virtual void vload(version_type &t) =  0;
virtual void vload(object_id_type &t) =  0;
virtual void vload(class_id_type &t) =  0;
virtual void vload(class_id_optional_type &t) = 0;
virtual void vload(class_name_type &t) = 0;
virtual void vload(tracking_type &t) = 0;
protected:
BOOST_ARCHIVE_DECL basic_iarchive(unsigned int flags);
boost::archive::detail::helper_collection &
get_helper_collection(){
return *this;
}
public:
virtual BOOST_ARCHIVE_DECL ~basic_iarchive();
BOOST_ARCHIVE_DECL void next_object_pointer(void *t);
BOOST_ARCHIVE_DECL void register_basic_serializer(
const basic_iserializer & bis
);
BOOST_ARCHIVE_DECL void load_object(
void *t,
const basic_iserializer & bis
);
BOOST_ARCHIVE_DECL const basic_pointer_iserializer *
load_pointer(
void * & t,
const basic_pointer_iserializer * bpis_ptr,
const basic_pointer_iserializer * (*finder)(
const boost::serialization::extended_type_info & eti
)
);
BOOST_ARCHIVE_DECL void
set_library_version(boost::serialization::library_version_type archive_library_version);
BOOST_ARCHIVE_DECL boost::serialization::library_version_type
get_library_version() const;
BOOST_ARCHIVE_DECL unsigned int
get_flags() const;
BOOST_ARCHIVE_DECL void
reset_object_address(const void * new_address, const void * old_address);
BOOST_ARCHIVE_DECL void
delete_created_pointers();
};

} 
} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
