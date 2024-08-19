

#ifndef BOOST_BIMAP_DETAIL_GENERATE_VIEW_BINDER_HPP
#define BOOST_BIMAP_DETAIL_GENERATE_VIEW_BINDER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/multi_index/tag.hpp>


#define BOOST_BIMAP_GENERATE_MAP_VIEW_BINDER(                                 \
\
MAP_VIEW_TYPE                                                             \
\
)                                                                             \
\
template< class Tag, class BimapType >                                        \
struct map_view_bind                                                          \
{                                                                             \
typedef MAP_VIEW_TYPE                                                     \
<                                                                         \
Tag,                                                                  \
BimapType                                                             \
\
> type;                                                                   \
};




#define BOOST_BIMAP_GENERATE_SET_VIEW_BINDER(                                 \
\
SET_VIEW_TYPE                                                             \
\
)                                                                             \
\
template< class IndexType >                                                   \
struct set_view_bind                                                          \
{                                                                             \
typedef SET_VIEW_TYPE<IndexType> type;                                    \
};



#endif 
