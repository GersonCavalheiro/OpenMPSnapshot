#ifndef BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_GCC_X86_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_GCC_X86_HPP_INCLUDED


#include <boost/smart_ptr/detail/sp_obsolete.hpp>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using g++/x86 atomic_count")

#endif

BOOST_SP_OBSOLETE()

namespace boost
{

namespace detail
{

class atomic_count
{
public:

explicit atomic_count( long v ) : value_( static_cast< int >( v ) ) {}

long operator++()
{
return atomic_exchange_and_add( &value_, +1 ) + 1;
}

long operator--()
{
return atomic_exchange_and_add( &value_, -1 ) - 1;
}

operator long() const
{
return atomic_exchange_and_add( &value_, 0 );
}

private:

atomic_count(atomic_count const &);
atomic_count & operator=(atomic_count const &);

mutable int value_;

private:

static int atomic_exchange_and_add( int * pw, int dv )
{

int r;

__asm__ __volatile__
(
"lock\n\t"
"xadd %1, %0":
"+m"( *pw ), "=r"( r ): 
"1"( dv ): 
"memory", "cc" 
);

return r;
}
};

} 

} 

#endif 
