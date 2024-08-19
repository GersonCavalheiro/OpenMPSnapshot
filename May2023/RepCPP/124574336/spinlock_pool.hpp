#ifndef BOOST_SMART_PTR_DETAIL_SPINLOCK_POOL_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SPINLOCK_POOL_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>
#include <boost/smart_ptr/detail/spinlock.hpp>
#include <cstddef>

namespace boost
{

namespace detail
{

template< int M > class spinlock_pool
{
private:

static spinlock pool_[ 41 ];

public:

static spinlock & spinlock_for( void const * pv )
{
#if defined(__VMS) && __INITIAL_POINTER_SIZE == 64  
std::size_t i = reinterpret_cast< unsigned long long >( pv ) % 41;
#else  
std::size_t i = reinterpret_cast< std::size_t >( pv ) % 41;
#endif  
return pool_[ i ];
}

class scoped_lock
{
private:

spinlock & sp_;

scoped_lock( scoped_lock const & );
scoped_lock & operator=( scoped_lock const & );

public:

explicit scoped_lock( void const * pv ): sp_( spinlock_for( pv ) )
{
sp_.lock();
}

~scoped_lock()
{
sp_.unlock();
}
};
};

template< int M > spinlock spinlock_pool< M >::pool_[ 41 ] =
{
BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, 
BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, 
BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, 
BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, 
BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, 
BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, 
BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, 
BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT, 
BOOST_DETAIL_SPINLOCK_INIT
};

} 
} 

#endif 
