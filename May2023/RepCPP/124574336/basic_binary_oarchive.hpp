#ifndef BOOST_ARCHIVE_BASIC_BINARY_OARCHIVE_HPP
#define BOOST_ARCHIVE_BASIC_BINARY_OARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif






#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/integer.hpp>
#include <boost/integer_traits.hpp>

#include <boost/archive/detail/common_oarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/collection_size_type.hpp>
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
class BOOST_SYMBOL_VISIBLE basic_binary_oarchive :
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
typedef detail::common_oarchive<Archive> detail_common_oarchive;
template<class T>
void save_override(const T & t){
this->detail_common_oarchive::save_override(t);
}

BOOST_STATIC_ASSERT(sizeof(tracking_type) == sizeof(bool));
BOOST_STATIC_ASSERT(sizeof(class_id_type) == sizeof(int_least16_t));
BOOST_STATIC_ASSERT(sizeof(class_id_reference_type) == sizeof(int_least16_t));
BOOST_STATIC_ASSERT(sizeof(object_id_type) == sizeof(uint_least32_t));
BOOST_STATIC_ASSERT(sizeof(object_reference_type) == sizeof(uint_least32_t));

void save_override(const class_id_optional_type & ){}

#if 0
void save_override(const boost::archive::version_type & t){
library_version_type lvt = this->get_library_version();
if(boost::serialization::library_version_type(7) < lvt){
this->detail_common_oarchive::save_override(t);
}
else
if(boost::serialization::library_version_type(6) < lvt){
const boost::uint_least16_t x = t;
* this->This() << x;
}
else{
const unsigned int x = t;
* this->This() << x;
}
}
void save_override(const boost::serialization::item_version_type & t){
library_version_type lvt = this->get_library_version();
if(boost::serialization::library_version_type(7) < lvt){
this->detail_common_oarchive::save_override(t);
}
else
if(boost::serialization::library_version_type(6) < lvt){
const boost::uint_least16_t x = t;
* this->This() << x;
}
else{
const unsigned int x = t;
* this->This() << x;
}
}

void save_override(class_id_type & t){
library_version_type lvt = this->get_library_version();
if(boost::serialization::library_version_type(7) < lvt){
this->detail_common_oarchive::save_override(t);
}
else
if(boost::serialization::library_version_type(6) < lvt){
const boost::int_least16_t x = t;
* this->This() << x;
}
else{
const int x = t;
* this->This() << x;
}
}
void save_override(class_id_reference_type & t){
save_override(static_cast<class_id_type &>(t));
}

#endif

void save_override(const class_name_type & t){
const std::string s(t);
* this->This() << s;
}

#if 0
void save_override(const serialization::collection_size_type & t){
if (get_library_version() < boost::serialization::library_version_type(6)){
unsigned int x=0;
* this->This() >> x;
t = serialization::collection_size_type(x);
}
else{
* this->This() >> t;
}
}
#endif
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
init();

basic_binary_oarchive(unsigned int flags) :
detail::common_oarchive<Archive>(flags)
{}
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
