#ifndef BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_PT_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_PT_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_typeinfo_.hpp>
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <pthread.h>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using pthread_mutex sp_counted_base")

#endif

namespace boost
{

namespace detail
{

class BOOST_SYMBOL_VISIBLE sp_counted_base
{
private:

sp_counted_base( sp_counted_base const & );
sp_counted_base & operator= ( sp_counted_base const & );

boost::int_least32_t use_count_;        
boost::int_least32_t weak_count_;       

mutable pthread_mutex_t m_;

public:

sp_counted_base(): use_count_( 1 ), weak_count_( 1 )
{

#if defined(__hpux) && defined(_DECTHREADS_)
BOOST_VERIFY( pthread_mutex_init( &m_, pthread_mutexattr_default ) == 0 );
#else
BOOST_VERIFY( pthread_mutex_init( &m_, 0 ) == 0 );
#endif
}

virtual ~sp_counted_base() 
{
BOOST_VERIFY( pthread_mutex_destroy( &m_ ) == 0 );
}


virtual void dispose() = 0; 


virtual void destroy() 
{
delete this;
}

virtual void * get_deleter( sp_typeinfo_ const & ti ) = 0;
virtual void * get_local_deleter( sp_typeinfo_ const & ti ) = 0;
virtual void * get_untyped_deleter() = 0;

void add_ref_copy()
{
BOOST_VERIFY( pthread_mutex_lock( &m_ ) == 0 );
++use_count_;
BOOST_VERIFY( pthread_mutex_unlock( &m_ ) == 0 );
}

bool add_ref_lock() 
{
BOOST_VERIFY( pthread_mutex_lock( &m_ ) == 0 );
bool r = use_count_ == 0? false: ( ++use_count_, true );
BOOST_VERIFY( pthread_mutex_unlock( &m_ ) == 0 );
return r;
}

void release() 
{
BOOST_VERIFY( pthread_mutex_lock( &m_ ) == 0 );
boost::int_least32_t new_use_count = --use_count_;
BOOST_VERIFY( pthread_mutex_unlock( &m_ ) == 0 );

if( new_use_count == 0 )
{
dispose();
weak_release();
}
}

void weak_add_ref() 
{
BOOST_VERIFY( pthread_mutex_lock( &m_ ) == 0 );
++weak_count_;
BOOST_VERIFY( pthread_mutex_unlock( &m_ ) == 0 );
}

void weak_release() 
{
BOOST_VERIFY( pthread_mutex_lock( &m_ ) == 0 );
boost::int_least32_t new_weak_count = --weak_count_;
BOOST_VERIFY( pthread_mutex_unlock( &m_ ) == 0 );

if( new_weak_count == 0 )
{
destroy();
}
}

long use_count() const 
{
BOOST_VERIFY( pthread_mutex_lock( &m_ ) == 0 );
boost::int_least32_t r = use_count_;
BOOST_VERIFY( pthread_mutex_unlock( &m_ ) == 0 );

return r;
}
};

} 

} 

#endif  
