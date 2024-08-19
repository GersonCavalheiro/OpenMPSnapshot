#ifndef BOOST_ARCHIVE_DETAIL_INTERFACE_IARCHIVE_HPP
#define BOOST_ARCHIVE_DETAIL_INTERFACE_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif



#include <cstddef> 
#include <boost/cstdint.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/detail/iserializer.hpp>
#include <boost/archive/detail/helper_collection.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace archive {
namespace detail {

class basic_pointer_iserializer;

template<class Archive>
class interface_iarchive
{
protected:
interface_iarchive() {}
public:
typedef mpl::bool_<true> is_loading;
typedef mpl::bool_<false> is_saving;

Archive * This(){
return static_cast<Archive *>(this);
}

template<class T>
const basic_pointer_iserializer *
register_type(T * = NULL){
const basic_pointer_iserializer & bpis =
boost::serialization::singleton<
pointer_iserializer<Archive, T>
>::get_const_instance();
this->This()->register_basic_serializer(bpis.get_basic_serializer());
return & bpis;
}
template<class Helper>
Helper &
get_helper(void * const id = 0){
helper_collection & hc = this->This()->get_helper_collection();
return hc.template find_helper<Helper>(id);
}

template<class T>
Archive & operator>>(T & t){
this->This()->load_override(t);
return * this->This();
}

template<class T>
Archive & operator&(T & t){
return *(this->This()) >> t;
}
};

} 
} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
