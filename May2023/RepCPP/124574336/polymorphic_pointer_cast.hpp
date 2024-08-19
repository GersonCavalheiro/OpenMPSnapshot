

#ifndef BOOST_CONVERSION_POLYMORPHIC_POINTER_CAST_HPP
#define BOOST_CONVERSION_POLYMORPHIC_POINTER_CAST_HPP

# include <boost/config.hpp>
# include <boost/assert.hpp>
# include <boost/pointer_cast.hpp>
# include <boost/throw_exception.hpp>
# include <boost/utility/declval.hpp>
# ifdef BOOST_NO_CXX11_DECLTYPE
#   include <boost/typeof/typeof.hpp>
# endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

namespace boost
{





namespace detail
{
template <typename Target, typename Source>
struct dynamic_pointer_cast_result
{
#ifdef BOOST_NO_CXX11_DECLTYPE
BOOST_TYPEOF_NESTED_TYPEDEF_TPL(nested, dynamic_pointer_cast<Target>(boost::declval<Source>()))
typedef typename nested::type type;
#else
typedef decltype(dynamic_pointer_cast<Target>(boost::declval<Source>())) type;
#endif
};
}

template <typename Target, typename Source>
inline typename detail::dynamic_pointer_cast_result<Target, Source>::type
polymorphic_pointer_downcast (const Source& x)
{
BOOST_ASSERT(dynamic_pointer_cast<Target> (x) == x);
return static_pointer_cast<Target> (x);
}

template <typename Target, typename Source>
inline typename detail::dynamic_pointer_cast_result<Target, Source>::type
polymorphic_pointer_cast (const Source& x)
{
typename detail::dynamic_pointer_cast_result<Target, Source>::type tmp
= dynamic_pointer_cast<Target> (x);
if ( !tmp ) boost::throw_exception( std::bad_cast() );

return tmp;
}

} 

#endif  
