#ifndef BOOST_ARCHIVE_TEXT_WOARCHIVE_HPP
#define BOOST_ARCHIVE_TEXT_WOARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>

#ifdef BOOST_NO_STD_WSTREAMBUF
#error "wide char i/o not supported on this platform"
#else

#include <ostream>
#include <cstddef> 

#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif

#include <boost/archive/detail/auto_link_warchive.hpp>
#include <boost/archive/basic_text_oprimitive.hpp>
#include <boost/archive/basic_text_oarchive.hpp>
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
class BOOST_SYMBOL_VISIBLE text_woarchive_impl :
public basic_text_oprimitive<std::wostream>,
public basic_text_oarchive<Archive>
{
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
#if BOOST_WORKAROUND(BOOST_MSVC, < 1500)
friend detail::interface_oarchive<Archive>;
friend basic_text_oarchive<Archive>;
friend save_access;
#else
friend class detail::interface_oarchive<Archive>;
friend class basic_text_oarchive<Archive>;
friend class save_access;
#endif
#endif
template<class T>
void save(const T & t){
this->newtoken();
basic_text_oprimitive<std::wostream>::save(t);
}
void save(const version_type & t){
save(static_cast<unsigned int>(t));
}
void save(const boost::serialization::item_version_type & t){
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
text_woarchive_impl(std::wostream & os, unsigned int flags) :
basic_text_oprimitive<std::wostream>(
os,
0 != (flags & no_codecvt)
),
basic_text_oarchive<Archive>(flags)
{}
public:
void save_binary(const void *address, std::size_t count){
put(static_cast<wchar_t>('\n'));
this->end_preamble();
#if ! defined(__MWERKS__)
this->basic_text_oprimitive<std::wostream>::save_binary(
#else
this->basic_text_oprimitive::save_binary(
#endif
address,
count
);
put(static_cast<wchar_t>('\n'));
this->delimiter = this->none;
}

};


class BOOST_SYMBOL_VISIBLE text_woarchive :
public text_woarchive_impl<text_woarchive>
{
public:
text_woarchive(std::wostream & os, unsigned int flags = 0) :
text_woarchive_impl<text_woarchive>(os, flags)
{
if(0 == (flags & no_header))
init();
}
~text_woarchive() BOOST_OVERRIDE {}
};

} 
} 

BOOST_SERIALIZATION_REGISTER_ARCHIVE(boost::archive::text_woarchive)

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
#endif 
