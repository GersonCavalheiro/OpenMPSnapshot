#ifndef BOOST_ARCHIVE_XML_IARCHIVE_HPP
#define BOOST_ARCHIVE_XML_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <istream>

#include <boost/scoped_ptr.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/basic_text_iprimitive.hpp>
#include <boost/archive/basic_xml_iarchive.hpp>
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
template<class Archive> class interface_iarchive;
} 

template<class CharType>
class basic_xml_grammar;
typedef basic_xml_grammar<char> xml_grammar;

template<class Archive>
class BOOST_SYMBOL_VISIBLE xml_iarchive_impl :
public basic_text_iprimitive<std::istream>,
public basic_xml_iarchive<Archive>
{
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
friend class detail::interface_iarchive<Archive>;
friend class basic_xml_iarchive<Archive>;
friend class load_access;
#endif
boost::scoped_ptr<xml_grammar> gimpl;

std::istream & get_is(){
return is;
}
template<class T>
void load(T & t){
basic_text_iprimitive<std::istream>::load(t);
}
void
load(version_type & t){
unsigned int v;
load(v);
t = version_type(v);
}
void
load(boost::serialization::item_version_type & t){
unsigned int v;
load(v);
t = boost::serialization::item_version_type(v);
}
BOOST_ARCHIVE_DECL void
load(char * t);
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
BOOST_ARCHIVE_DECL void
load(wchar_t * t);
#endif
BOOST_ARCHIVE_DECL void
load(std::string &s);
#ifndef BOOST_NO_STD_WSTRING
BOOST_ARCHIVE_DECL void
load(std::wstring &ws);
#endif
template<class T>
void load_override(T & t){
basic_xml_iarchive<Archive>::load_override(t);
}
BOOST_ARCHIVE_DECL void
load_override(class_name_type & t);
BOOST_ARCHIVE_DECL void
init();
BOOST_ARCHIVE_DECL
xml_iarchive_impl(std::istream & is, unsigned int flags);
BOOST_ARCHIVE_DECL
~xml_iarchive_impl() BOOST_OVERRIDE;
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

class BOOST_SYMBOL_VISIBLE xml_iarchive :
public xml_iarchive_impl<xml_iarchive>{
public:
xml_iarchive(std::istream & is, unsigned int flags = 0) :
xml_iarchive_impl<xml_iarchive>(is, flags)
{
if(0 == (flags & no_header))
init();
}
~xml_iarchive() BOOST_OVERRIDE {}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::xml_iarchive)

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
