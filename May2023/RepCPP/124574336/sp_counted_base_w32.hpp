#ifndef BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_W32_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_W32_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_interlocked.hpp>
#include <boost/smart_ptr/detail/sp_typeinfo_.hpp>
#include <boost/config/workaround.hpp>
#include <boost/config.hpp>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using Win32 sp_counted_base")

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

long use_count_;        
long weak_count_;       

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
BOOST_SP_INTERLOCKED_INCREMENT( &use_count_ );
}

bool add_ref_lock() 
{
for( ;; )
{
long tmp = static_cast< long const volatile& >( use_count_ );
if( tmp == 0 ) return false;

#if defined( BOOST_MSVC ) && BOOST_WORKAROUND( BOOST_MSVC, == 1200 )


long tmp2 = tmp + 1;
if( BOOST_SP_INTERLOCKED_COMPARE_EXCHANGE( &use_count_, tmp2, tmp ) == tmp2 - 1 ) return true;

#else

if( BOOST_SP_INTERLOCKED_COMPARE_EXCHANGE( &use_count_, tmp + 1, tmp ) == tmp ) return true;

#endif
}
}

void release() 
{
if( BOOST_SP_INTERLOCKED_DECREMENT( &use_count_ ) == 0 )
{
dispose();
weak_release();
}
}

void weak_add_ref() 
{
BOOST_SP_INTERLOCKED_INCREMENT( &weak_count_ );
}

void weak_release() 
{
if( BOOST_SP_INTERLOCKED_DECREMENT( &weak_count_ ) == 0 )
{
destroy();
}
}

long use_count() const 
{
return static_cast<long const volatile &>( use_count_ );
}
};

} 

} 

#endif  
