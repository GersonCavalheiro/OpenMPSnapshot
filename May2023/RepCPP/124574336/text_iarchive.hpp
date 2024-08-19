#ifndef BOOST_ARCHIVE_TEXT_IARCHIVE_HPP
#define BOOST_ARCHIVE_TEXT_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <istream>

#include <boost/config.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/basic_text_iprimitive.hpp>
#include <boost/archive/basic_text_iarchive.hpp>
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

template<class Archive>
class BOOST_SYMBOL_VISIBLE text_iarchive_impl :
public basic_text_iprimitive<std::istream>,
public basic_text_iarchive<Archive>
{
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
friend class detail::interface_iarchive<Archive>;
friend class load_access;
#endif
template<class T>
void load(T & t){
basic_text_iprimitive<std::istream>::load(t);
}
void load(version_type & t){
unsigned int v;
load(v);
t = version_type(v);
}
void load(boost::serialization::item_version_type & t){
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
basic_text_iarchive<Archive>::load_override(t);
}
BOOST_ARCHIVE_DECL void
load_override(class_name_type & t);
BOOST_ARCHIVE_DECL void
init();
BOOST_ARCHIVE_DECL
text_iarchive_impl(std::istream & is, unsigned int flags);
~text_iarchive_impl() BOOST_OVERRIDE {}
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

class BOOST_SYMBOL_VISIBLE text_iarchive :
public text_iarchive_impl<text_iarchive>{
public:
text_iarchive(std::istream & is_, unsigned int flags = 0) :
text_iarchive_impl<text_iarchive>(is_, flags)
{
if(0 == (flags & no_header))
init();
}
~text_iarchive() BOOST_OVERRIDE {}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::text_iarchive)

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
