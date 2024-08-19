

#ifndef BOOST_BIMAP_RELATION_SUPPORT_GET_PAIR_FUNCTOR_HPP
#define BOOST_BIMAP_RELATION_SUPPORT_GET_PAIR_FUNCTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/relation/support/pair_by.hpp>

namespace boost {
namespace bimaps {
namespace relation {
namespace support {



template< class Tag, class Relation >
struct get_pair_functor
{
BOOST_DEDUCED_TYPENAME result_of::pair_by<Tag,Relation>::type
operator()( Relation & r ) const
{
return pair_by<Tag>(r);
}

BOOST_DEDUCED_TYPENAME result_of::pair_by<Tag,const Relation>::type
operator()( const Relation & r ) const
{
return pair_by<Tag>(r);
}
};




template< class Relation >
struct get_above_view_functor
{
BOOST_DEDUCED_TYPENAME Relation::above_view &
operator()( Relation & r ) const
{
return r.get_view();
}

const BOOST_DEDUCED_TYPENAME Relation::above_view &
operator()( const Relation & r ) const
{
return r.get_view();
}
};

} 
} 
} 
} 


#endif 

