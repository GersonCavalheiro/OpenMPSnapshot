
#ifndef BOOST_CONTAINER_DETAIL_POOL_COMMON_HPP
#define BOOST_CONTAINER_DETAIL_POOL_COMMON_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>

#include <boost/intrusive/slist.hpp>

namespace boost {
namespace container {
namespace dtl {

template<class VoidPointer>
struct node_slist
{
typedef typename bi::make_slist_base_hook
<bi::void_pointer<VoidPointer>, bi::link_mode<bi::normal_link> >::type slist_hook_t;

typedef slist_hook_t node_t;

typedef typename bi::make_slist
<node_t, bi::linear<true>, bi::cache_last<true>, bi::base_hook<slist_hook_t> >::type node_slist_t;
};

template<class T>
struct is_stateless_segment_manager
{
static const bool value = false;
};

}  
}  
}  

#include <boost/container/detail/config_end.hpp>

#endif   
