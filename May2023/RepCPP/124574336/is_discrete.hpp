
#ifndef BOOST_ICL_TYPE_TRAITS_IS_DISCRETE_HPP_JOFA_100410
#define BOOST_ICL_TYPE_TRAITS_IS_DISCRETE_HPP_JOFA_100410

#include <string>
#include <boost/config.hpp> 
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>

#ifdef BOOST_MSVC 
#pragma warning(push)
#pragma warning(disable:4913) 
#endif                        

#include <boost/detail/is_incrementable.hpp>

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/icl/type_traits/rep_type_of.hpp>
#include <boost/icl/type_traits/is_numeric.hpp>

namespace boost{ namespace icl
{
template <class Type> struct is_discrete
{
typedef is_discrete type;
BOOST_STATIC_CONSTANT(bool, 
value = 
(mpl::and_
< 
boost::detail::is_incrementable<Type>
, mpl::or_
< 
mpl::and_
<
mpl::not_<has_rep_type<Type> >
, is_non_floating_point<Type>
>
, mpl::and_
<
has_rep_type<Type>
, is_discrete<typename rep_type_of<Type>::type>
>
>
>::value
)
);
};

}} 

#endif


