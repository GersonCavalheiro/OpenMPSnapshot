

#ifndef BOOST_BIMAP_DETAIL_GENERATE_RELATION_BINDER_HPP
#define BOOST_BIMAP_DETAIL_GENERATE_RELATION_BINDER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/mpl/apply.hpp>


#define BOOST_BIMAP_GENERATE_RELATION_BINDER_0CP(                             \
\
SET_TYPE_OF                                                           \
)                                                                         \
\
template< class Relation >                                                \
struct bind_to                                                            \
{                                                                         \
typedef SET_TYPE_OF<Relation> type;                                   \
\
};





#define BOOST_BIMAP_GENERATE_RELATION_BINDER_1CP(                             \
\
SET_TYPE_OF,                                                          \
CP1                                                                   \
)                                                                         \
\
template< class Relation >                                                \
struct bind_to                                                            \
{                                                                         \
typedef SET_TYPE_OF                                                   \
<                                                                     \
Relation,                                                         \
BOOST_DEDUCED_TYPENAME mpl::apply<CP1,                            \
BOOST_DEDUCED_TYPENAME Relation::storage_base >::type         \
\
> type;                                                               \
\
};





#define BOOST_BIMAP_GENERATE_RELATION_BINDER_2CP(                             \
\
SET_TYPE_OF,                                                          \
CP1,                                                                  \
CP2                                                                   \
)                                                                         \
\
template< class Relation >                                                \
struct bind_to                                                            \
{                                                                         \
typedef SET_TYPE_OF                                                   \
<                                                                     \
Relation,                                                         \
BOOST_DEDUCED_TYPENAME mpl::apply<CP1,                            \
BOOST_DEDUCED_TYPENAME Relation::storage_base >::type,        \
BOOST_DEDUCED_TYPENAME mpl::apply<CP2,                            \
BOOST_DEDUCED_TYPENAME Relation::storage_base >::type         \
\
> type;                                                               \
\
};




#endif 
