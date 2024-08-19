#ifndef BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_SNC_PS3_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_SNC_PS3_HPP_INCLUDED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_typeinfo_.hpp>
#include <boost/smart_ptr/detail/sp_obsolete.hpp>
#include <boost/config.hpp>
#include <inttypes.h> 

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using PS3 sp_counted_base")

#endif

BOOST_SP_OBSOLETE()

namespace boost
{

namespace detail
{

inline uint32_t compare_and_swap( uint32_t * dest_, uint32_t compare_, uint32_t swap_ )
{
return __builtin_cellAtomicCompareAndSwap32(dest_,compare_,swap_);
}

inline uint32_t atomic_fetch_and_add( uint32_t * pw, uint32_t dv )
{

for( ;; )
{
uint32_t r = *pw;

if( __builtin_expect((compare_and_swap(pw, r, r + dv) == r), 1) )
{
return r;
}
}
}

inline void atomic_increment( uint32_t * pw )
{
(void) __builtin_cellAtomicIncr32( pw );
}

inline uint32_t atomic_decrement( uint32_t * pw )
{
return __builtin_cellAtomicDecr32( pw );
}

inline uint32_t atomic_conditional_increment( uint32_t * pw )
{

for( ;; )
{
uint32_t r = *pw;

if( r == 0 )
{
return r;
}

if( __builtin_expect( ( compare_and_swap( pw, r, r + 1 ) == r ), 1 ) )
{
return r;
}
}    
}

class BOOST_SYMBOL_VISIBLE sp_counted_base
{
private:

sp_counted_base( sp_counted_base const & );
sp_counted_base & operator= ( sp_counted_base const & );

uint32_t use_count_;        
uint32_t weak_count_;       

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
if( atomic_decrement( &use_count_ ) == 1 )
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
if( atomic_decrement( &weak_count_ ) == 1 )
{
destroy();
}
}

long use_count() const 
{
return const_cast< uint32_t const volatile & >( use_count_ );
}
};

} 

} 

#endif  
