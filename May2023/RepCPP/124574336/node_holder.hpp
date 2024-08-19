
#ifndef BOOST_INTRUSIVE_DETAIL_NODE_HOLDER_HPP
#define BOOST_INTRUSIVE_DETAIL_NODE_HOLDER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

template<class Node, class Tag, unsigned int>
struct node_holder
:  public Node
{};

}  
}  

#endif 
