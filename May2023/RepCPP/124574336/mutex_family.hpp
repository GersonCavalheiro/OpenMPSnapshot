
#ifndef BOOST_INTERPROCESS_MUTEX_FAMILY_HPP
#define BOOST_INTERPROCESS_MUTEX_FAMILY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_recursive_mutex.hpp>
#include <boost/interprocess/sync/null_mutex.hpp>


namespace boost {

namespace interprocess {

struct mutex_family
{
typedef boost::interprocess::interprocess_mutex                 mutex_type;
typedef boost::interprocess::interprocess_recursive_mutex       recursive_mutex_type;
};

struct null_mutex_family
{
typedef boost::interprocess::null_mutex                   mutex_type;
typedef boost::interprocess::null_mutex                   recursive_mutex_type;
};

}  

}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   


