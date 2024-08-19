#ifndef BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_SPIN_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_SPIN_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_typeinfo_.hpp>
#include <boost/smart_ptr/detail/spinlock_pool.hpp>
#include <boost/config.hpp>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using spinlock-based sp_counted_base")

#endif

namespace boost
{

namespace detail
{

inline int atomic_exchange_and_add( int * pw, int dv )
{
spinlock_pool<1>::scoped_lock lock( pw );

int r = *pw;
*pw += dv;
return r;
}

inline void atomic_increment( int * pw )
{
spinlock_pool<1>::scoped_lock lock( pw );
++*pw;
}

inline int atomic_conditional_increment( int * pw )
{
spinlock_pool<1>::scoped_lock lock( pw );

int rv = *pw;
if( rv != 0 ) ++*pw;
return rv;
}

class BOOST_SYMBOL_VISIBLE sp_counted_base
{
private:

sp_counted_base( sp_counted_base const & );
sp_counted_base & operator= ( sp_counted_base const & );

int use_count_;        
int weak_count_;       

public:

sp_counted_base(): use_count_( 1 ), weak_count_( 1 )
{
}

virtual ~sp_counted_base() 
{
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
atomic_increment( &use_count_ );
}

bool add_ref_lock() 
{
return atomic_conditional_increment( &use_count_ ) != 0;
}

void release() 
{
if( atomic_exchange_and_add( &use_count_, -1 ) == 1 )
{
dispose();
weak_release();
}
}

void weak_add_ref() 
{
atomic_increment( &weak_count_ );
}

void weak_release() 
{
if( atomic_exchange_and_add( &weak_count_, -1 ) == 1 )
{
destroy();
}
}

long use_count() const 
{
spinlock_pool<1>::scoped_lock lock( &use_count_ );
return use_count_;
}
};

} 

} 

#endif  
