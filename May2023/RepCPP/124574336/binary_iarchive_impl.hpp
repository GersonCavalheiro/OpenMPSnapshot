#ifndef BOOST_ARCHIVE_BINARY_IARCHIVE_IMPL_HPP
#define BOOST_ARCHIVE_BINARY_IARCHIVE_IMPL_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <istream>
#include <boost/archive/basic_binary_iprimitive.hpp>
#include <boost/archive/basic_binary_iarchive.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {

namespace detail {
template<class Archive> class interface_iarchive;
} 

template<class Archive, class Elem, class Tr>
class BOOST_SYMBOL_VISIBLE binary_iarchive_impl :
public basic_binary_iprimitive<Archive, Elem, Tr>,
public basic_binary_iarchive<Archive>
{
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
#if BOOST_WORKAROUND(BOOST_MSVC, < 1500)
friend detail::interface_iarchive<Archive>;
friend basic_binary_iarchive<Archive>;
friend load_access;
#else
friend class detail::interface_iarchive<Archive>;
friend class basic_binary_iarchive<Archive>;
friend class load_access;
#endif
#endif
template<class T>
void load_override(T & t){
this->basic_binary_iarchive<Archive>::load_override(t);
}
void init(unsigned int flags){
if(0 != (flags & no_header)){
return;
}
#if ! defined(__MWERKS__)
this->basic_binary_iarchive<Archive>::init();
this->basic_binary_iprimitive<Archive, Elem, Tr>::init();
#else
basic_binary_iarchive<Archive>::init();
basic_binary_iprimitive<Archive, Elem, Tr>::init();
#endif
}
binary_iarchive_impl(
std::basic_streambuf<Elem, Tr> & bsb,
unsigned int flags
) :
basic_binary_iprimitive<Archive, Elem, Tr>(
bsb,
0 != (flags & no_codecvt)
),
basic_binary_iarchive<Archive>(flags)
{}
binary_iarchive_impl(
std::basic_istream<Elem, Tr> & is,
unsigned int flags
) :
basic_binary_iprimitive<Archive, Elem, Tr>(
* is.rdbuf(),
0 != (flags & no_codecvt)
),
basic_binary_iarchive<Archive>(flags)
{}
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
