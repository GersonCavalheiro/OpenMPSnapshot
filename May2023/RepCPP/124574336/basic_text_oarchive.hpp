#ifndef BOOST_ARCHIVE_BASIC_TEXT_OARCHIVE_HPP
#define BOOST_ARCHIVE_BASIC_TEXT_OARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/archive/detail/common_oarchive.hpp>
#include <boost/serialization/string.hpp>

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
class BOOST_SYMBOL_VISIBLE basic_text_oarchive :
public detail::common_oarchive<Archive>
{
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
#if BOOST_WORKAROUND(BOOST_MSVC, < 1500)
friend detail::interface_oarchive<Archive>;
#else
friend class detail::interface_oarchive<Archive>;
#endif
#endif

enum {
none,
eol,
space
} delimiter;

BOOST_ARCHIVE_OR_WARCHIVE_DECL void
newtoken();

void newline(){
delimiter = eol;
}

typedef detail::common_oarchive<Archive> detail_common_oarchive;
template<class T>
void save_override(T & t){
this->detail_common_oarchive::save_override(t);
}

void save_override(const object_id_type & t){
this->This()->newline();
this->detail_common_oarchive::save_override(t);
}

void save_override(const class_id_optional_type & ){}

void save_override(const class_name_type & t){
const std::string s(t);
* this->This() << s;
}

BOOST_ARCHIVE_OR_WARCHIVE_DECL void
init();

basic_text_oarchive(unsigned int flags) :
detail::common_oarchive<Archive>(flags),
delimiter(none)
{}
~basic_text_oarchive() BOOST_OVERRIDE {}
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
