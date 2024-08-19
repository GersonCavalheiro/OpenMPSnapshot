#ifndef BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_NT_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_NT_HPP_INCLUDED


#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using single-threaded, non-atomic atomic_count")

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
return ++value_;
}

long operator--()
{
return --value_;
}

operator long() const
{
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
