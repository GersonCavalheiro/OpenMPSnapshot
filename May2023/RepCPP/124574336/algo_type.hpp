
#ifndef BOOST_INTRUSIVE_DETAIL_ALGO_TYPE_HPP
#define BOOST_INTRUSIVE_DETAIL_ALGO_TYPE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

enum algo_types
{
CircularListAlgorithms,
CircularSListAlgorithms,
LinearSListAlgorithms,
CommonSListAlgorithms,
BsTreeAlgorithms,
RbTreeAlgorithms,
AvlTreeAlgorithms,
SgTreeAlgorithms,
SplayTreeAlgorithms,
TreapAlgorithms,
UnorderedAlgorithms,
UnorderedCircularSlistAlgorithms,
AnyAlgorithm
};

template<algo_types AlgoType, class NodeTraits>
struct get_algo;

template<algo_types AlgoType, class ValueTraits, class NodePtrCompare, class ExtraChecker>
struct get_node_checker;

} 
} 

#endif 
