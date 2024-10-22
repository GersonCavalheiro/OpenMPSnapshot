#ifndef BOOST_ARCHIVE_XML_WOARCHIVE_HPP
#define BOOST_ARCHIVE_XML_WOARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifdef BOOST_NO_STD_WSTREAMBUF
#error "wide char i/o not supported on this platform"
#else
#include <cstddef> 
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif

#include <ostream>

#include <boost/archive/detail/auto_link_warchive.hpp>
#include <boost/archive/basic_text_oprimitive.hpp>
#include <boost/archive/basic_xml_oarchive.hpp>
#include <boost/archive/detail/register_archive.hpp>
#include <boost/serialization/item_version_type.hpp>

#include <boost/archive/detail/abi_prefix.hpp> 

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {

namespace detail {
template<class Archive> class interface_oarchive;
} 

template<class Archive>
class BOOST_SYMBOL_VISIBLE xml_woarchive_impl :
public basic_text_oprimitive<std::wostream>,
public basic_xml_oarchive<Archive>
{
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
friend class detail::interface_oarchive<Archive>;
friend class basic_xml_oarchive<Archive>;
friend class save_access;
#endif
template<class T>
void
save(const T & t){
basic_text_oprimitive<std::wostream>::save(t);
}
void
save(const version_type & t){
save(static_cast<unsigned int>(t));
}
void
save(const boost::serialization::item_version_type & t){
save(static_cast<unsigned int>(t));
}
BOOST_WARCHIVE_DECL void
save(const char * t);
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
BOOST_WARCHIVE_DECL void
save(const wchar_t * t);
#endif
BOOST_WARCHIVE_DECL void
save(const std::string &s);
#ifndef BOOST_NO_STD_WSTRING
BOOST_WARCHIVE_DECL void
save(const std::wstring &ws);
#endif
BOOST_WARCHIVE_DECL
xml_woarchive_impl(std::wostream & os, unsigned int flags);
BOOST_WARCHIVE_DECL
~xml_woarchive_impl() BOOST_OVERRIDE;
public:
BOOST_WARCHIVE_DECL void
save_binary(const void *address, std::size_t count);

};


class BOOST_SYMBOL_VISIBLE xml_woarchive :
public xml_woarchive_impl<xml_woarchive>
{
public:
xml_woarchive(std::wostream & os, unsigned int flags = 0) :
xml_woarchive_impl<xml_woarchive>(os, flags)
{
if(0 == (flags & no_header))
init();
}
~xml_woarchive() BOOST_OVERRIDE {}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::xml_woarchive)

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
#endif 
