#ifndef BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_WIN32_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_WIN32_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_interlocked.hpp>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using Win32 atomic_count")

#endif

namespace boost
{

namespace detail
{

class atomic_count
{
public:

explicit atomic_count( long v ): value_( v )
{
}

long operator++()
{
return BOOST_SP_INTERLOCKED_INCREMENT( &value_ );
}

long operator--()
{
return BOOST_SP_INTERLOCKED_DECREMENT( &value_ );
}

operator long() const
{
return static_cast<long const volatile &>( value_ );
}

private:

atomic_count( atomic_count const & );
atomic_count & operator=( atomic_count const & );

long value_;
};

} 

} 

#endif 
