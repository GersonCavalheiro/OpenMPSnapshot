#ifndef BOOST_ARCHIVE_TEXT_WIARCHIVE_HPP
#define BOOST_ARCHIVE_TEXT_WIARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifdef BOOST_NO_STD_WSTREAMBUF
#error "wide char i/o not supported on this platform"
#else

#include <istream>

#include <boost/archive/detail/auto_link_warchive.hpp>
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
class BOOST_SYMBOL_VISIBLE text_wiarchive_impl :
public basic_text_iprimitive<std::wistream>,
public basic_text_iarchive<Archive>
{
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
#if BOOST_WORKAROUND(BOOST_MSVC, < 1500)
friend detail::interface_iarchive<Archive>;
friend load_access;
#else
friend class detail::interface_iarchive<Archive>;
friend class load_access;
#endif
#endif
template<class T>
void load(T & t){
basic_text_iprimitive<std::wistream>::load(t);
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
BOOST_WARCHIVE_DECL void
load(char * t);
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
BOOST_WARCHIVE_DECL void
load(wchar_t * t);
#endif
BOOST_WARCHIVE_DECL void
load(std::string &s);
#ifndef BOOST_NO_STD_WSTRING
BOOST_WARCHIVE_DECL void
load(std::wstring &ws);
#endif
template<class T>
void load_override(T & t){
basic_text_iarchive<Archive>::load_override(t);
}
BOOST_WARCHIVE_DECL
text_wiarchive_impl(std::wistream & is, unsigned int flags);
~text_wiarchive_impl() BOOST_OVERRIDE {}
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

class BOOST_SYMBOL_VISIBLE text_wiarchive :
public text_wiarchive_impl<text_wiarchive>{
public:
text_wiarchive(std::wistream & is, unsigned int flags = 0) :
text_wiarchive_impl<text_wiarchive>(is, flags)
{
if(0 == (flags & no_header))
init();
}
~text_wiarchive() BOOST_OVERRIDE {}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::text_wiarchive)

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
#endif 
