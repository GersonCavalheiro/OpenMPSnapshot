#ifndef BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_STD_ATOMIC_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_STD_ATOMIC_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_typeinfo_.hpp>
#include <boost/smart_ptr/detail/sp_noexcept.hpp>
#include <boost/config.hpp>
#include <atomic>
#include <cstdint>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using std::atomic sp_counted_base")

#endif

namespace boost
{

namespace detail
{

inline void atomic_increment( std::atomic_int_least32_t * pw ) BOOST_SP_NOEXCEPT
{
pw->fetch_add( 1, std::memory_order_relaxed );
}

inline std::int_least32_t atomic_decrement( std::atomic_int_least32_t * pw ) BOOST_SP_NOEXCEPT
{
return pw->fetch_sub( 1, std::memory_order_acq_rel );
}

inline std::int_least32_t atomic_conditional_increment( std::atomic_int_least32_t * pw ) BOOST_SP_NOEXCEPT
{

std::int_least32_t r = pw->load( std::memory_order_relaxed );

for( ;; )
{
if( r == 0 )
{
return r;
}

if( pw->compare_exchange_weak( r, r + 1, std::memory_order_relaxed, std::memory_order_relaxed ) )
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

std::atomic_int_least32_t use_count_;   
std::atomic_int_least32_t weak_count_;  

public:

sp_counted_base() BOOST_SP_NOEXCEPT: use_count_( 1 ), weak_count_( 1 )
{
}

virtual ~sp_counted_base() 
{
}


virtual void dispose() BOOST_SP_NOEXCEPT = 0;


virtual void destroy() BOOST_SP_NOEXCEPT
{
delete this;
}

virtual void * get_deleter( sp_typeinfo_ const & ti ) BOOST_SP_NOEXCEPT = 0;
virtual void * get_local_deleter( sp_typeinfo_ const & ti ) BOOST_SP_NOEXCEPT = 0;
virtual void * get_untyped_deleter() BOOST_SP_NOEXCEPT = 0;

void add_ref_copy() BOOST_SP_NOEXCEPT
{
atomic_increment( &use_count_ );
}

bool add_ref_lock() BOOST_SP_NOEXCEPT 
{
return atomic_conditional_increment( &use_count_ ) != 0;
}

void release() BOOST_SP_NOEXCEPT
{
if( atomic_decrement( &use_count_ ) == 1 )
{
dispose();
weak_release();
}
}

void weak_add_ref() BOOST_SP_NOEXCEPT
{
atomic_increment( &weak_count_ );
}

void weak_release() BOOST_SP_NOEXCEPT
{
if( atomic_decrement( &weak_count_ ) == 1 )
{
destroy();
}
}

long use_count() const BOOST_SP_NOEXCEPT
{
return use_count_.load( std::memory_order_acquire );
}
};

} 

} 

#endif  
