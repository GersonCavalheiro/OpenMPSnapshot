


#ifndef BOOST_BIMAP_DETAIL_GENERATE_INDEX_BINDER_HPP
#define BOOST_BIMAP_DETAIL_GENERATE_INDEX_BINDER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/multi_index/tag.hpp>



#define BOOST_BIMAP_GENERATE_INDEX_BINDER_0CP(                                \
\
MULTI_INDEX_TYPE                                                          \
\
)                                                                             \
\
template< class KeyExtractor, class Tag >                                     \
struct index_bind                                                             \
{                                                                             \
typedef MULTI_INDEX_TYPE                                                  \
<                                                                         \
multi_index::tag< Tag >,                                              \
KeyExtractor                                                          \
\
> type;                                                                   \
};





#define BOOST_BIMAP_GENERATE_INDEX_BINDER_1CP(                                \
\
MULTI_INDEX_TYPE,                                                         \
CONFIG_PARAMETER                                                          \
\
)                                                                             \
\
template< class KeyExtractor, class Tag >                                     \
struct index_bind                                                             \
{                                                                             \
typedef MULTI_INDEX_TYPE                                                  \
<                                                                         \
multi_index::tag< Tag >,                                              \
KeyExtractor,                                                         \
CONFIG_PARAMETER                                                      \
\
> type;                                                                   \
};






#define BOOST_BIMAP_GENERATE_INDEX_BINDER_2CP(                                \
\
MULTI_INDEX_TYPE,                                                         \
CONFIG_PARAMETER_1,                                                       \
CONFIG_PARAMETER_2                                                        \
)                                                                             \
\
template< class KeyExtractor, class Tag >                                     \
struct index_bind                                                             \
{                                                                             \
typedef MULTI_INDEX_TYPE                                                  \
<                                                                         \
multi_index::tag< Tag >,                                              \
KeyExtractor,                                                         \
CONFIG_PARAMETER_1,                                                   \
CONFIG_PARAMETER_2                                                    \
\
> type;                                                                   \
\
};





#define BOOST_BIMAP_GENERATE_INDEX_BINDER_0CP_NO_EXTRACTOR(                   \
\
MULTI_INDEX_TYPE                                                          \
\
)                                                                             \
\
template< class KeyExtractor, class Tag >                                     \
struct index_bind                                                             \
{                                                                             \
typedef MULTI_INDEX_TYPE< multi_index::tag< Tag > > type;                 \
};





#define BOOST_BIMAP_GENERATE_INDEX_BINDER_FAKE                                \
\
template< class KeyExtractor, class Tag >                                     \
struct index_bind                                                             \
{                                                                             \
typedef void type;                                                        \
};                                                                            \


#endif 
