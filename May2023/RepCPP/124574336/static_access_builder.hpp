


#ifndef BOOST_BIMAP_RELATION_DETAIL_STATIC_ACCESS_BUILDER_HPP
#define BOOST_BIMAP_RELATION_DETAIL_STATIC_ACCESS_BUILDER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/relation/support/is_tag_of_member_at.hpp>
#include <boost/bimap/detail/debug/static_error.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/preprocessor/cat.hpp>







#define BOOST_BIMAP_SYMMETRIC_STATIC_ACCESS_BUILDER(                          \
\
NAME,                                                                 \
SYMMETRIC_TYPE,                                                       \
LEFT_BODY,                                                            \
RIGHT_BODY                                                            \
)                                                                         \
\
template                                                                  \
<                                                                         \
class Tag,                                                            \
class SYMMETRIC_TYPE,                                                 \
class Enable = void                                                   \
>                                                                         \
struct NAME                                                               \
{                                                                         \
BOOST_BIMAP_STATIC_ERROR(                                             \
BOOST_PP_CAT(NAME,_FAILURE),                                      \
(SYMMETRIC_TYPE,Tag)                                              \
);                                                                    \
};                                                                        \
\
template< class Tag, class SYMMETRIC_TYPE >                               \
struct NAME                                                               \
<                                                                         \
Tag, SYMMETRIC_TYPE,                                                  \
BOOST_DEDUCED_TYPENAME enable_if                                      \
<                                                                     \
::boost::bimaps::relation::support::is_tag_of_member_at_left      \
<                                                                 \
Tag,                                                          \
SYMMETRIC_TYPE                                                \
>                                                                 \
\
>::type                                                               \
>                                                                         \
{                                                                         \
LEFT_BODY;                                                            \
};                                                                        \
\
template< class Tag, class SYMMETRIC_TYPE >                               \
struct NAME                                                               \
<                                                                         \
Tag, SYMMETRIC_TYPE,                                                  \
BOOST_DEDUCED_TYPENAME enable_if                                      \
<                                                                     \
::boost::bimaps::relation::support::is_tag_of_member_at_right     \
<                                                                 \
Tag,                                                          \
SYMMETRIC_TYPE                                                \
>                                                                 \
\
>::type                                                               \
>                                                                         \
{                                                                         \
RIGHT_BODY;                                                           \
};



#endif 


