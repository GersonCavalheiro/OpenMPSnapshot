#ifndef BOOST_ARCHIVE_BASIC_BINARY_IARCHIVE_HPP
#define BOOST_ARCHIVE_BASIC_BINARY_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/archive/basic_archive.hpp>
#include <boost/archive/detail/common_iarchive.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/library_version_type.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/integer_traits.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace archive {

namespace detail {
template<class Archive> class interface_iarchive;
} 

template<class Archive>
class BOOST_SYMBOL_VISIBLE basic_binary_iarchive :
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

BOOST_STATIC_ASSERT(sizeof(class_id_type) == sizeof(int_least16_t));
BOOST_STATIC_ASSERT(sizeof(class_id_reference_type) == sizeof(int_least16_t));
BOOST_STATIC_ASSERT(sizeof(object_id_type) == sizeof(uint_least32_t));
BOOST_STATIC_ASSERT(sizeof(object_reference_type) == sizeof(uint_least32_t));

void load_override(class_id_optional_type & ){}

void load_override(tracking_type & t, int ){
boost::serialization::library_version_type lv = this->get_library_version();
if(boost::serialization::library_version_type(6) < lv){
int_least8_t x=0;
* this->This() >> x;
t = boost::archive::tracking_type(x);
}
else{
bool x=0;
* this->This() >> x;
t = boost::archive::tracking_type(x);
}
}
void load_override(class_id_type & t){
boost::serialization::library_version_type lv = this->get_library_version();

if(boost::serialization::library_version_type (7) < lv){
this->detail_common_iarchive::load_override(t);
}
else{
int_least16_t x=0;
* this->This() >> x;
t = boost::archive::class_id_type(x);
}
}
void load_override(class_id_reference_type & t){
load_override(static_cast<class_id_type &>(t));
}

void load_override(version_type & t){
boost::serialization::library_version_type  lv = this->get_library_version();
if(boost::serialization::library_version_type(7) < lv){
this->detail_common_iarchive::load_override(t);
}
else
if(boost::serialization::library_version_type(6) < lv){
uint_least8_t x=0;
* this->This() >> x;
t = boost::archive::version_type(x);
}
else
if(boost::serialization::library_version_type(5) < lv){
uint_least16_t x=0;
* this->This() >> x;
t = boost::archive::version_type(x);
}
else
if(boost::serialization::library_version_type(2) < lv){
unsigned char x=0;
* this->This() >> x;
t = version_type(x);
}
else{
unsigned int x=0;
* this->This() >> x;
t = boost::archive::version_type(x);
}
}

void load_override(boost::serialization::item_version_type & t){
boost::serialization::library_version_type lv = this->get_library_version();
if(boost::serialization::library_version_type(6) < lv){
this->detail_common_iarchive::load_override(t);
}
else
if(boost::serialization::library_version_type(6) < lv){
uint_least16_t x=0;
* this->This() >> x;
t = boost::serialization::item_version_type(x);
}
else{
unsigned int x=0;
* this->This() >> x;
t = boost::serialization::item_version_type(x);
}
}

void load_override(serialization::collection_size_type & t){
if(boost::serialization::library_version_type(5) < this->get_library_version()){
this->detail_common_iarchive::load_override(t);
}
else{
unsigned int x=0;
* this->This() >> x;
t = serialization::collection_size_type(x);
}
}

BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load_override(class_name_type & t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
init();

basic_binary_iarchive(unsigned int flags) :
detail::common_iarchive<Archive>(flags)
{}
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
