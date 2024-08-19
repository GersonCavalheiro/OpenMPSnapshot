
#ifndef BOOST_RANGE_ADAPTOR_ARGUMENT_FWD_HPP
#define BOOST_RANGE_ADAPTOR_ARGUMENT_FWD_HPP

#include <boost/config.hpp>

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable : 4512) 
#endif

namespace boost
{
namespace range_detail
{  
template< class T >
struct holder
{
T val;
holder( T t ) : val(t)
{ }
};

template< class T >
struct holder2
{
T val1, val2;
holder2( T t, T u ) : val1(t), val2(u)
{ }
};

template< template<class> class Holder >
struct forwarder
{
template< class T >
Holder<T> operator()( T t ) const
{
return Holder<T>(t);
}
};

template< template<class> class Holder >
struct forwarder2
{
template< class T >
Holder<T> operator()( T t, T u ) const
{
return Holder<T>(t,u);
}
};

template< template<class,class> class Holder >
struct forwarder2TU
{
template< class T, class U >
Holder<T, U> operator()( T t, U u ) const
{
return Holder<T, U>(t, u);
}
};


} 

}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif
