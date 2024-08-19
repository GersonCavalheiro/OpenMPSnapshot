#ifndef BOOST_ARCHIVE_DETAIL_POLYMORPHIC_OARCHIVE_ROUTE_HPP
#define BOOST_ARCHIVE_DETAIL_POLYMORPHIC_OARCHIVE_ROUTE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <string>
#include <ostream>
#include <cstddef> 

#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif

#include <boost/cstdint.hpp>
#include <boost/integer_traits.hpp>
#include <boost/archive/polymorphic_oarchive.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace serialization {
class extended_type_info;
} 
namespace archive {
namespace detail{

class basic_oserializer;
class basic_pointer_oserializer;

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

template<class ArchiveImplementation>
class polymorphic_oarchive_route :
public polymorphic_oarchive,
public  ArchiveImplementation
{
private:
void save_object(
const void *x,
const detail::basic_oserializer & bos
) BOOST_OVERRIDE {
ArchiveImplementation::save_object(x, bos);
}
void save_pointer(
const void * t,
const detail::basic_pointer_oserializer * bpos_ptr
) BOOST_OVERRIDE {
ArchiveImplementation::save_pointer(t, bpos_ptr);
}
void save_null_pointer() BOOST_OVERRIDE {
ArchiveImplementation::save_null_pointer();
}
void save(const bool t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const char t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const signed char t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const unsigned char t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
#ifndef BOOST_NO_CWCHAR
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
void save(const wchar_t t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
#endif
#endif
void save(const short t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const unsigned short t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const int t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const unsigned int t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const long t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const unsigned long t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
#if defined(BOOST_HAS_LONG_LONG)
void save(const boost::long_long_type t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const boost::ulong_long_type t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
#elif defined(BOOST_HAS_MS_INT64)
void save(const boost::int64_t t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const boost::uint64_t t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
#endif
void save(const float t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const double t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
void save(const std::string & t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
#ifndef BOOST_NO_STD_WSTRING
void save(const std::wstring & t) BOOST_OVERRIDE {
ArchiveImplementation::save(t);
}
#endif
boost::serialization::library_version_type get_library_version() const BOOST_OVERRIDE {
return ArchiveImplementation::get_library_version();
}
unsigned int get_flags() const BOOST_OVERRIDE {
return ArchiveImplementation::get_flags();
}
void save_binary(const void * t, std::size_t size) BOOST_OVERRIDE {
ArchiveImplementation::save_binary(t, size);
}
void save_start(const char * name) BOOST_OVERRIDE {
ArchiveImplementation::save_start(name);
}
void save_end(const char * name) BOOST_OVERRIDE {
ArchiveImplementation::save_end(name);
}
void end_preamble() BOOST_OVERRIDE {
ArchiveImplementation::end_preamble();
}
void register_basic_serializer(const detail::basic_oserializer & bos) BOOST_OVERRIDE {
ArchiveImplementation::register_basic_serializer(bos);
}
helper_collection &
get_helper_collection() BOOST_OVERRIDE {
return ArchiveImplementation::get_helper_collection();
}
public:
typedef mpl::bool_<false> is_loading;
typedef mpl::bool_<true> is_saving;
template<class T>
polymorphic_oarchive & operator<<(T & t){
return polymorphic_oarchive::operator<<(t);
}
template<class T>
polymorphic_oarchive & operator&(T & t){
return polymorphic_oarchive::operator&(t);
}
template<class T>
const basic_pointer_oserializer *
register_type(T * t = NULL){
return ArchiveImplementation::register_type(t);
}
template <class _Elem, class _Tr>
polymorphic_oarchive_route(
std::basic_ostream<_Elem, _Tr> & os,
unsigned int flags = 0
) :
ArchiveImplementation(os, flags)
{}
~polymorphic_oarchive_route() BOOST_OVERRIDE {}
};

} 
} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
