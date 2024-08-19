#ifndef BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_GCC_X86_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_GCC_X86_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_typeinfo_.hpp>
#include <boost/smart_ptr/detail/sp_obsolete.hpp>
#include <boost/config.hpp>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using g++/x86 sp_counted_base")

#endif

BOOST_SP_OBSOLETE()

namespace boost
{

namespace detail
{

inline int atomic_exchange_and_add( int * pw, int dv )
{

int r;

__asm__ __volatile__
(
"lock\n\t"
"xadd %1, %0":
"=m"( *pw ), "=r"( r ): 
"m"( *pw ), "1"( dv ): 
"memory", "cc" 
);

return r;
}

inline void atomic_increment( int * pw )
{

__asm__
(
"lock\n\t"
"incl %0":
"=m"( *pw ): 
"m"( *pw ): 
"cc" 
);
}

inline int atomic_conditional_increment( int * pw )
{

int rv, tmp;

__asm__
(
"movl %0, %%eax\n\t"
"0:\n\t"
"test %%eax, %%eax\n\t"
"je 1f\n\t"
"movl %%eax, %2\n\t"
"incl %2\n\t"
"lock\n\t"
"cmpxchgl %2, %0\n\t"
"jne 0b\n\t"
"1:":
"=m"( *pw ), "=&a"( rv ), "=&r"( tmp ): 
"m"( *pw ): 
"cc" 
);

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
return static_cast<int const volatile &>( use_count_ );
}
};

} 

} 

#endif  
