
#ifndef BOOST_INTERPROCESS_WINDOWS_NAMED_CONDITION_HPP
#define BOOST_INTERPROCESS_WINDOWS_NAMED_CONDITION_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/sync/windows/named_condition_any.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail {

typedef windows_named_condition_any windows_named_condition;

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
