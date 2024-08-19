#ifndef BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_AIX_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_AIX_HPP_INCLUDED


#include <boost/smart_ptr/detail/sp_typeinfo_.hpp>
#include <boost/config.hpp>
#include <builtins.h>
#include <sys/atomic_op.h>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using AIX sp_counted_base")

#endif

namespace boost
{

namespace detail
{

inline void atomic_increment( int32_t* pw )
{

fetch_and_add( pw, 1 );
}

inline int32_t atomic_decrement( int32_t * pw )
{

int32_t originalValue;

__lwsync();
originalValue = fetch_and_add( pw, -1 );
__isync();

return (originalValue - 1);
}

inline int32_t atomic_conditional_increment( int32_t * pw )
{

int32_t tmp = fetch_and_add( pw, 0 );
for( ;; )
{
if( tmp == 0 ) return 0;
if( compare_and_swap( pw, &tmp, tmp + 1 ) ) return (tmp + 1);
}
}

class BOOST_SYMBOL_VISIBLE sp_counted_base
{
private:

sp_counted_base( sp_counted_base const & );
sp_counted_base & operator= ( sp_counted_base const & );

int32_t use_count_;        
int32_t weak_count_;       

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
if( atomic_decrement( &use_count_ ) == 0 )
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
if( atomic_decrement( &weak_count_ ) == 0 )
{
destroy();
}
}

long use_count() const 
{
return fetch_and_add( const_cast<int32_t*>(&use_count_), 0 );
}
};

} 

} 

#endif  
