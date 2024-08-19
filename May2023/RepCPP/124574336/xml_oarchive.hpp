#ifndef BOOST_ARCHIVE_XML_OARCHIVE_HPP
#define BOOST_ARCHIVE_XML_OARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <ostream>

#include <cstddef> 
#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif

#include <boost/archive/detail/auto_link_archive.hpp>
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
class BOOST_SYMBOL_VISIBLE xml_oarchive_impl :
public basic_text_oprimitive<std::ostream>,
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
void save(const T & t){
basic_text_oprimitive<std::ostream>::save(t);
}
void
save(const version_type & t){
save(static_cast<unsigned int>(t));
}
void
save(const boost::serialization::item_version_type & t){
save(static_cast<unsigned int>(t));
}
BOOST_ARCHIVE_DECL void
save(const char * t);
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
BOOST_ARCHIVE_DECL void
save(const wchar_t * t);
#endif
BOOST_ARCHIVE_DECL void
save(const std::string &s);
#ifndef BOOST_NO_STD_WSTRING
BOOST_ARCHIVE_DECL void
save(const std::wstring &ws);
#endif
BOOST_ARCHIVE_DECL
xml_oarchive_impl(std::ostream & os, unsigned int flags);
BOOST_ARCHIVE_DECL
~xml_oarchive_impl() BOOST_OVERRIDE;
public:
BOOST_ARCHIVE_DECL
void save_binary(const void *address, std::size_t count);
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {


class BOOST_SYMBOL_VISIBLE xml_oarchive :
public xml_oarchive_impl<xml_oarchive>
{
public:
xml_oarchive(std::ostream & os, unsigned int flags = 0) :
xml_oarchive_impl<xml_oarchive>(os, flags)
{
if(0 == (flags & no_header))
init();
}
~xml_oarchive() BOOST_OVERRIDE {}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::xml_oarchive)

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
