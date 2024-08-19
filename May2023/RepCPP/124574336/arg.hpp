#ifndef BOOST_BIND_ARG_HPP_INCLUDED
#define BOOST_BIND_ARG_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>
#include <boost/is_placeholder.hpp>

namespace boost
{

template<bool Eq> struct _arg_eq
{
};

template<> struct _arg_eq<true>
{
typedef void type;
};

template< int I > struct arg
{
BOOST_CONSTEXPR arg()
{
}

template< class T > BOOST_CONSTEXPR arg( T const & , typename _arg_eq< I == is_placeholder<T>::value >::type * = 0 )
{
}
};

template< int I > BOOST_CONSTEXPR bool operator==( arg<I> const &, arg<I> const & )
{
return true;
}

#if !defined( BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION )

template< int I > struct is_placeholder< arg<I> >
{
enum _vt { value = I };
};

template< int I > struct is_placeholder< arg<I> (*) () >
{
enum _vt { value = I };
};

#endif

} 

#endif 
