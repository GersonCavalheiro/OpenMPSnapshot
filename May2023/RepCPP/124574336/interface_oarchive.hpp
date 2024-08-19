#ifndef BOOST_ARCHIVE_DETAIL_INTERFACE_OARCHIVE_HPP
#define BOOST_ARCHIVE_DETAIL_INTERFACE_OARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif



#include <cstddef> 
#include <boost/cstdint.hpp>
#include <boost/mpl/bool.hpp>

#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/detail/oserializer.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

#include <boost/serialization/singleton.hpp>

namespace boost {
namespace archive {
namespace detail {

class basic_pointer_oserializer;

template<class Archive>
class interface_oarchive
{
protected:
interface_oarchive() {}
public:
typedef mpl::bool_<false> is_loading;
typedef mpl::bool_<true> is_saving;

Archive * This(){
return static_cast<Archive *>(this);
}

template<class T>
const basic_pointer_oserializer *
register_type(const T * = NULL){
const basic_pointer_oserializer & bpos =
boost::serialization::singleton<
pointer_oserializer<Archive, T>
>::get_const_instance();
this->This()->register_basic_serializer(bpos.get_basic_serializer());
return & bpos;
}

template<class Helper>
Helper &
get_helper(void * const id = 0){
helper_collection & hc = this->This()->get_helper_collection();
return hc.template find_helper<Helper>(id);
}

template<class T>
Archive & operator<<(const T & t){
this->This()->save_override(t);
return * this->This();
}

template<class T>
Archive & operator&(const T & t){
return * this ->This() << t;
}
};

} 
} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
