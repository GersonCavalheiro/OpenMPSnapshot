
#ifndef BOOST_INTERPROCESS_DETAIL_NODE_TOOLS_HPP
#define BOOST_INTERPROCESS_DETAIL_NODE_TOOLS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/intrusive/slist.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail {


template<class VoidPointer>
struct node_slist
{
typedef typename bi::make_slist_base_hook
<bi::void_pointer<VoidPointer>, bi::link_mode<bi::normal_link> >::type slist_hook_t;

struct node_t
:  public slist_hook_t
{};

typedef typename bi::make_slist
<node_t, bi::linear<true>, bi::base_hook<slist_hook_t> >::type node_slist_t;
};

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
