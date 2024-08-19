#ifndef BOOST_ARCHIVE_BASIC_TEXT_IARCHIVE_HPP
#define BOOST_ARCHIVE_BASIC_TEXT_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/archive/detail/common_iarchive.hpp>

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
class BOOST_SYMBOL_VISIBLE basic_text_iarchive :
public detail::common_iarchive<Archive>
{
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
#if BOOST_WORKAROUND(BOOST_MSVC, < 1500)
friend detail::interface_iarchive<Archive>;
#else
friend class detail::interface_iarchive<Archive>;
#endif
#endif
typedef detail::common_iarchive<Archive> detail_common_iarchive;
template<class T>
void load_override(T & t){
this->detail_common_iarchive::load_override(t);
}
void load_override(class_id_optional_type & ){}

BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load_override(class_name_type & t);

BOOST_ARCHIVE_OR_WARCHIVE_DECL void
init();

basic_text_iarchive(unsigned int flags) :
detail::common_iarchive<Archive>(flags)
{}
~basic_text_iarchive() BOOST_OVERRIDE {}
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
