#ifndef BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_SPIN_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_SPIN_HPP_INCLUDED


#include <boost/smart_ptr/detail/spinlock_pool.hpp>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using spinlock-based atomic_count")

#endif

namespace boost
{

namespace detail
{

class atomic_count
{
private:

public:

explicit atomic_count( long v ): value_( v )
{
}

long operator++()
{
spinlock_pool<0>::scoped_lock lock( &value_ );
return ++value_;
}

long operator--()
{
spinlock_pool<0>::scoped_lock lock( &value_ );
return --value_;
}

operator long() const
{
spinlock_pool<0>::scoped_lock lock( &value_ );
return value_;
}

private:

atomic_count(atomic_count const &);
atomic_count & operator=(atomic_count const &);

long value_;
};

} 

} 

#endif 
