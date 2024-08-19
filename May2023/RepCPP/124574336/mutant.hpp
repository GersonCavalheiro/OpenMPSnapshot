

#ifndef BOOST_BIMAP_RELATION_DETAIL_MUTANT_HPP
#define BOOST_BIMAP_RELATION_DETAIL_MUTANT_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/detail/debug/static_error.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/utility/addressof.hpp>
#include <boost/mpl/not.hpp>
#include <boost/utility/enable_if.hpp>

namespace boost {
namespace bimaps {
namespace relation {


namespace detail {




template< class View, class Type >
BOOST_DEDUCED_TYPENAME enable_if< mpl::not_< is_const< Type > >,

View&

>::type mutate( Type & m )
{
BOOST_MPL_ASSERT((
::boost::mpl::contains<BOOST_DEDUCED_TYPENAME Type::mutant_views,View>
));
return *reinterpret_cast< View* >(boost::addressof(m));
}

template< class View, class Type >
BOOST_DEDUCED_TYPENAME enable_if< is_const< Type >,

const View&

>::type mutate( Type & m )
{
BOOST_MPL_ASSERT((
::boost::mpl::contains<BOOST_DEDUCED_TYPENAME Type::mutant_views,View> 
));
return *reinterpret_cast< const View* >(boost::addressof(m));
}


} 
} 
} 
} 

#endif 

