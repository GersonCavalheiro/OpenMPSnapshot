#ifndef BOOST_INTERPROCESS_WINDOWS_RECURSIVE_NAMED_MUTEX_HPP
#define BOOST_INTERPROCESS_WINDOWS_RECURSIVE_NAMED_MUTEX_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/sync/windows/named_mutex.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail {


class windows_named_recursive_mutex
: public windows_named_mutex
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

windows_named_recursive_mutex();
windows_named_recursive_mutex(const windows_named_mutex &);
windows_named_recursive_mutex &operator=(const windows_named_mutex &);
#endif   

public:
windows_named_recursive_mutex(create_only_t, const char *name, const permissions &perm = permissions())
: windows_named_mutex(create_only_t(), name, perm)
{}

windows_named_recursive_mutex(open_or_create_t, const char *name, const permissions &perm = permissions())
: windows_named_mutex(open_or_create_t(), name, perm)
{}

windows_named_recursive_mutex(open_only_t, const char *name)
: windows_named_mutex(open_only_t(), name)
{}
};

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
