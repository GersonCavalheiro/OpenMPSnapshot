#ifndef BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_GCC_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_GCC_HPP_INCLUDED


#if __GNUC__ * 100 + __GNUC_MINOR__ >= 402
# include <ext/atomicity.h> 
#else 
# include <bits/atomicity.h>
#endif

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using libstdc++ atomic_count")

#endif

namespace boost
{

namespace detail
{

#if defined(__GLIBCXX__) 

using __gnu_cxx::__atomic_add;
using __gnu_cxx::__exchange_and_add;

#endif

class atomic_count
{
public:

explicit atomic_count( long v ) : value_( v ) {}

long operator++()
{
return __exchange_and_add( &value_, +1 ) + 1;
}

long operator--()
{
return __exchange_and_add( &value_, -1 ) - 1;
}

operator long() const
{
return __exchange_and_add( &value_, 0 );
}

private:

atomic_count(atomic_count const &);
atomic_count & operator=(atomic_count const &);

mutable _Atomic_word value_;
};

} 

} 

#endif 
