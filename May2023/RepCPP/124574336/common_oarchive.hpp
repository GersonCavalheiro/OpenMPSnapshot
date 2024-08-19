#ifndef BOOST_ARCHIVE_DETAIL_COMMON_OARCHIVE_HPP
#define BOOST_ARCHIVE_DETAIL_COMMON_OARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>

#include <boost/archive/detail/basic_oarchive.hpp>
#include <boost/archive/detail/interface_oarchive.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {
namespace detail {

template<class Archive>

class BOOST_SYMBOL_VISIBLE common_oarchive :
public basic_oarchive,
public interface_oarchive<Archive>
{
friend class interface_oarchive<Archive>;
friend class basic_oarchive;
private:
void vsave(const version_type t) BOOST_OVERRIDE {
* this->This() << t;
}
void vsave(const object_id_type t) BOOST_OVERRIDE {
* this->This() << t;
}
void vsave(const object_reference_type t) BOOST_OVERRIDE {
* this->This() << t;
}
void vsave(const class_id_type t) BOOST_OVERRIDE {
* this->This() << t;
}
void vsave(const class_id_reference_type t) BOOST_OVERRIDE {
* this->This() << t;
}
void vsave(const class_id_optional_type t) BOOST_OVERRIDE {
* this->This() << t;
}
void vsave(const class_name_type & t) BOOST_OVERRIDE {
* this->This() << t;
}
void vsave(const tracking_type t) BOOST_OVERRIDE {
* this->This() << t;
}
protected:
template<class T>
void save_override(T & t){
archive::save(* this->This(), t);
}
void save_start(const char * ){}
void save_end(const char * ){}
common_oarchive(unsigned int flags = 0) :
basic_oarchive(flags),
interface_oarchive<Archive>()
{}
};

} 
} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
