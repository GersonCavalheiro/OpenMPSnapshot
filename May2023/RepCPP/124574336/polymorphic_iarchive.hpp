#ifndef BOOST_ARCHIVE_POLYMORPHIC_IARCHIVE_HPP
#define BOOST_ARCHIVE_POLYMORPHIC_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <cstddef> 
#include <climits> 
#include <string>

#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif

#include <boost/cstdint.hpp>

#include <boost/archive/detail/iserializer.hpp>
#include <boost/archive/detail/interface_iarchive.hpp>
#include <boost/serialization/library_version_type.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/archive/detail/register_archive.hpp>

#include <boost/archive/detail/decl.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace serialization {
class extended_type_info;
} 
namespace archive {
namespace detail {
class basic_iarchive;
class basic_iserializer;
}

class polymorphic_iarchive;

class BOOST_SYMBOL_VISIBLE polymorphic_iarchive_impl :
public detail::interface_iarchive<polymorphic_iarchive>
{
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
friend class detail::interface_iarchive<polymorphic_iarchive>;
friend class load_access;
#endif
virtual void load(bool & t) = 0;

virtual void load(char & t) = 0;
virtual void load(signed char & t) = 0;
virtual void load(unsigned char & t) = 0;
#ifndef BOOST_NO_CWCHAR
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
virtual void load(wchar_t & t) = 0;
#endif
#endif
virtual void load(short & t) = 0;
virtual void load(unsigned short & t) = 0;
virtual void load(int & t) = 0;
virtual void load(unsigned int & t) = 0;
virtual void load(long & t) = 0;
virtual void load(unsigned long & t) = 0;

#if defined(BOOST_HAS_LONG_LONG)
virtual void load(boost::long_long_type & t) = 0;
virtual void load(boost::ulong_long_type & t) = 0;
#elif defined(BOOST_HAS_MS_INT64)
virtual void load(__int64 & t) = 0;
virtual void load(unsigned __int64 & t) = 0;
#endif

virtual void load(float & t) = 0;
virtual void load(double & t) = 0;

virtual void load(std::string & t) = 0;
#ifndef BOOST_NO_STD_WSTRING
virtual void load(std::wstring & t) = 0;
#endif

virtual void load_start(const char * name) = 0;
virtual void load_end(const char * name) = 0;
virtual void register_basic_serializer(const detail::basic_iserializer & bis) = 0;
virtual detail::helper_collection & get_helper_collection() = 0;

template<class T>
void load_override(T & t)
{
archive::load(* this->This(), t);
}
template<class T>
void load_override(
const boost::serialization::nvp< T > & t
){
load_start(t.name());
archive::load(* this->This(), t.value());
load_end(t.name());
}
protected:
virtual ~polymorphic_iarchive_impl() {}
public:
virtual void set_library_version(
boost::serialization::library_version_type archive_library_version
) = 0;
virtual boost::serialization::library_version_type get_library_version() const = 0;
virtual unsigned int get_flags() const = 0;
virtual void delete_created_pointers() = 0;
virtual void reset_object_address(
const void * new_address,
const void * old_address
) = 0;

virtual void load_binary(void * t, std::size_t size) = 0;

virtual void load_object(
void *t,
const detail::basic_iserializer & bis
) = 0;
virtual const detail::basic_pointer_iserializer * load_pointer(
void * & t,
const detail::basic_pointer_iserializer * bpis_ptr,
const detail::basic_pointer_iserializer * (*finder)(
const boost::serialization::extended_type_info & type
)
) = 0;
};

} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE polymorphic_iarchive :
public polymorphic_iarchive_impl
{
public:
~polymorphic_iarchive() BOOST_OVERRIDE {}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::polymorphic_iarchive)

#endif 
