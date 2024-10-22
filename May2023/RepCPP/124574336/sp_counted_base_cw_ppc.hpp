#ifndef BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_CW_PPC_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_CW_PPC_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_typeinfo_.hpp>
#include <boost/smart_ptr/detail/sp_obsolete.hpp>
#include <boost/config.hpp>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using CodeWarrior/PowerPC sp_counted_base")

#endif

BOOST_SP_OBSOLETE()

namespace boost
{

namespace detail
{

inline void atomic_increment( register long * pw )
{
register int a;

asm
{
loop:

lwarx   a, 0, pw
addi    a, a, 1
stwcx.  a, 0, pw
bne-    loop
}
}

inline long atomic_decrement( register long * pw )
{
register int a;

asm
{
sync

loop:

lwarx   a, 0, pw
addi    a, a, -1
stwcx.  a, 0, pw
bne-    loop

isync
}

return a;
}

inline long atomic_conditional_increment( register long * pw )
{
register int a;

asm
{
loop:

lwarx   a, 0, pw
cmpwi   a, 0
beq     store

addi    a, a, 1

store:

stwcx.  a, 0, pw
bne-    loop
}

return a;
}

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
return static_cast<long const volatile &>( use_count_ );
}
};

} 

} 

#endif  
