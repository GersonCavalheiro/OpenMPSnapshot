#ifndef BOOST_ARCHIVE_BINARY_OARCHIVE_IMPL_HPP
#define BOOST_ARCHIVE_BINARY_OARCHIVE_IMPL_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <ostream>
#include <boost/config.hpp>
#include <boost/archive/basic_binary_oprimitive.hpp>
#include <boost/archive/basic_binary_oarchive.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {

namespace detail {
template<class Archive> class interface_oarchive;
} 

template<class Archive, class Elem, class Tr>
class BOOST_SYMBOL_VISIBLE binary_oarchive_impl :
public basic_binary_oprimitive<Archive, Elem, Tr>,
public basic_binary_oarchive<Archive>
{
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
#if BOOST_WORKAROUND(BOOST_MSVC, < 1500)
friend detail::interface_oarchive<Archive>;
friend basic_binary_oarchive<Archive>;
friend save_access;
#else
friend class detail::interface_oarchive<Archive>;
friend class basic_binary_oarchive<Archive>;
friend class save_access;
#endif
#endif
template<class T>
void save_override(T & t){
this->basic_binary_oarchive<Archive>::save_override(t);
}
void init(unsigned int flags) {
if(0 != (flags & no_header)){
return;
}
#if ! defined(__MWERKS__)
this->basic_binary_oarchive<Archive>::init();
this->basic_binary_oprimitive<Archive, Elem, Tr>::init();
#else
basic_binary_oarchive<Archive>::init();
basic_binary_oprimitive<Archive, Elem, Tr>::init();
#endif
}
binary_oarchive_impl(
std::basic_streambuf<Elem, Tr> & bsb,
unsigned int flags
) :
basic_binary_oprimitive<Archive, Elem, Tr>(
bsb,
0 != (flags & no_codecvt)
),
basic_binary_oarchive<Archive>(flags)
{}
binary_oarchive_impl(
std::basic_ostream<Elem, Tr> & os,
unsigned int flags
) :
basic_binary_oprimitive<Archive, Elem, Tr>(
* os.rdbuf(),
0 != (flags & no_codecvt)
),
basic_binary_oarchive<Archive>(flags)
{}
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
