

#ifndef BOOST_BIMAP_RELATION_ACCESS_BUILDER_HPP
#define BOOST_BIMAP_RELATION_ACCESS_BUILDER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/relation/support/member_with_tag.hpp>
#include <boost/bimap/relation/member_at.hpp>
#include <boost/call_traits.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/not.hpp>
#include <boost/utility/enable_if.hpp>





#define BOOST_BIMAP_SYMMETRIC_ACCESS_RESULT_OF_BUILDER(                       \
\
NAME,                                                                 \
METAFUNCTION_BASE                                                     \
)                                                                         \
\
namespace result_of {                                                     \
\
template< class Tag, class SymmetricType >                                \
struct NAME                                                               \
{                                                                         \
typedef BOOST_DEDUCED_TYPENAME METAFUNCTION_BASE                      \
<                                                                     \
Tag,SymmetricType                                                 \
\
>::type value_type;                                                   \
\
typedef BOOST_DEDUCED_TYPENAME mpl::if_< is_const<SymmetricType>,     \
\
BOOST_DEDUCED_TYPENAME call_traits<value_type>::const_reference,  \
\
BOOST_DEDUCED_TYPENAME call_traits<value_type>::reference         \
\
>::type type;                                                         \
};                                                                        \
\
}








#define BOOST_BIMAP_SYMMETRIC_ACCESS_IMPLEMENTATION_BUILDER(                  \
\
NAME,                                                                 \
TP_SYMMETRIC,                                                         \
PARAMETER_NAME,                                                       \
LEFT_BODY,                                                            \
RIGHT_BODY                                                            \
)                                                                         \
\
namespace detail {                                                        \
\
\
\
template< class TP_SYMMETRIC >                                            \
BOOST_DEDUCED_TYPENAME result_of::NAME                                    \
<                                                                         \
::boost::bimaps::relation::member_at::left,TP_SYMMETRIC               \
\
>::type                                                                   \
\
NAME( ::boost::bimaps::relation::member_at::left,                         \
TP_SYMMETRIC & PARAMETER_NAME )                             \
{                                                                         \
LEFT_BODY;                                                            \
}                                                                         \
\
template< class TP_SYMMETRIC >                                            \
BOOST_DEDUCED_TYPENAME result_of::NAME                                    \
<                                                                         \
::boost::bimaps::relation::member_at::right,TP_SYMMETRIC              \
\
>::type                                                                   \
\
NAME( ::boost::bimaps::relation::member_at::right,                        \
TP_SYMMETRIC & PARAMETER_NAME )                             \
{                                                                         \
RIGHT_BODY;                                                           \
}                                                                         \
\
}






#define BOOST_BIMAP_SYMMETRIC_ACCESS_INTERFACE_BUILDER(                       \
\
NAME                                                                  \
)                                                                         \
\
template< class Tag, class SymmetricType >                                \
BOOST_DEDUCED_TYPENAME result_of::NAME<Tag,SymmetricType>::type           \
NAME( SymmetricType & s )                                                 \
{                                                                         \
typedef BOOST_DEDUCED_TYPENAME ::boost::bimaps::relation::support::   \
member_with_tag                                                   \
<                                                                 \
Tag,SymmetricType                                             \
\
>::type member_at_tag;                                            \
\
return detail::NAME(member_at_tag(),s);                               \
}



#endif 

