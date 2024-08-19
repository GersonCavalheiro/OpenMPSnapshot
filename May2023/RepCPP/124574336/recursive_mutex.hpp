
#ifndef BOOST_INTERPROCESS_DETAIL_WINDOWS_RECURSIVE_MUTEX_HPP
#define BOOST_INTERPROCESS_DETAIL_WINDOWS_RECURSIVE_MUTEX_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/sync/windows/mutex.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail {

class windows_recursive_mutex
: public windows_mutex
{
windows_recursive_mutex(const windows_recursive_mutex &);
windows_recursive_mutex &operator=(const windows_recursive_mutex &);
public:
windows_recursive_mutex() : windows_mutex() {}
};

}  
}  
}  


#include <boost/interprocess/detail/config_end.hpp>

#endif   
