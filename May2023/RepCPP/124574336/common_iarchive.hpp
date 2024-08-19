#ifndef BOOST_ARCHIVE_DETAIL_COMMON_IARCHIVE_HPP
#define BOOST_ARCHIVE_DETAIL_COMMON_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>

#include <boost/archive/detail/basic_iarchive.hpp>
#include <boost/archive/detail/basic_pointer_iserializer.hpp>
#include <boost/archive/detail/interface_iarchive.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {
namespace detail {

class extended_type_info;

template<class Archive>
class BOOST_SYMBOL_VISIBLE common_iarchive :
public basic_iarchive,
public interface_iarchive<Archive>
{
friend class interface_iarchive<Archive>;
friend class basic_iarchive;
private:
void vload(version_type & t) BOOST_OVERRIDE {
* this->This() >> t;
}
void vload(object_id_type & t) BOOST_OVERRIDE {
* this->This() >> t;
}
void vload(class_id_type & t) BOOST_OVERRIDE {
* this->This() >> t;
}
void vload(class_id_optional_type & t) BOOST_OVERRIDE {
* this->This() >> t;
}
void vload(tracking_type & t) BOOST_OVERRIDE {
* this->This() >> t;
}
void vload(class_name_type &s) BOOST_OVERRIDE {
* this->This() >> s;
}
protected:
template<class T>
void load_override(T & t){
archive::load(* this->This(), t);
}
void load_start(const char * ){}
void load_end(const char * ){}
common_iarchive(unsigned int flags = 0) :
basic_iarchive(flags),
interface_iarchive<Archive>()
{}
};

} 
} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
