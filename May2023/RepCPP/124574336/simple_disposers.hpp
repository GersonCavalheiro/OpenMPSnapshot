
#ifndef BOOST_INTRUSIVE_DETAIL_SIMPLE_DISPOSERS_HPP
#define BOOST_INTRUSIVE_DETAIL_SIMPLE_DISPOSERS_HPP

#include <boost/intrusive/detail/workaround.hpp>

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {
namespace detail {

class null_disposer
{
public:
template <class Pointer>
void operator()(Pointer)
{}
};

template<class NodeAlgorithms>
class init_disposer
{
typedef typename NodeAlgorithms::node_ptr node_ptr;

public:
BOOST_INTRUSIVE_FORCEINLINE void operator()(const node_ptr & p)
{  NodeAlgorithms::init(p);   }
};

}  
}  
}  

#endif 
