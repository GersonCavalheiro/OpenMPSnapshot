#ifndef BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_SYNC_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_SYNC_HPP_INCLUDED


#include <boost/cstdint.hpp>

#if defined( __ia64__ ) && defined( __INTEL_COMPILER )
# include <ia64intrin.h>
#endif

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using __sync atomic_count")

#endif

namespace boost
{

namespace detail
{

class atomic_count
{
public:

explicit atomic_count( long v ): value_( static_cast< boost::int_least32_t >( v ) )
{
}

long operator++()
{
return __sync_add_and_fetch( &value_, 1 );
}

long operator--()
{
return __sync_add_and_fetch( &value_, -1 );
}

operator long() const
{
return __sync_fetch_and_add( &value_, 0 );
}

private:

atomic_count(atomic_count const &);
atomic_count & operator=(atomic_count const &);

mutable boost::int_least32_t value_;
};

} 

} 

#endif 
