#ifndef BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_ACC_IA64_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_COUNTED_BASE_ACC_IA64_HPP_INCLUDED


#include <boost/smart_ptr/detail/sp_typeinfo_.hpp>
#include <boost/smart_ptr/detail/sp_obsolete.hpp>
#include <boost/config.hpp>
#include <machine/sys/inline.h>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using HP aCC++/HP-UX/IA64 sp_counted_base")

#endif

BOOST_SP_OBSOLETE()

namespace boost
{

namespace detail
{

inline void atomic_increment( int * pw )
{

_Asm_fetchadd(_FASZ_W, _SEM_REL, pw, +1, _LDHINT_NONE);
} 

inline int atomic_decrement( int * pw )
{

int r = static_cast<int>(_Asm_fetchadd(_FASZ_W, _SEM_REL, pw, -1, _LDHINT_NONE));
if (1 == r)
{
_Asm_mf();
}

return r - 1;
}

inline int atomic_conditional_increment( int * pw )
{

int v = *pw;

for (;;)
{
if (0 == v)
{
return 0;
}

_Asm_mov_to_ar(_AREG_CCV,
v,
(_UP_CALL_FENCE | _UP_SYS_FENCE | _DOWN_CALL_FENCE | _DOWN_SYS_FENCE));
int r = static_cast<int>(_Asm_cmpxchg(_SZ_W, _SEM_ACQ, pw, v + 1, _LDHINT_NONE));
if (r == v)
{
return r + 1;
}

v = r;
}
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
return static_cast<int const volatile &>( use_count_ ); 
}
};

} 

} 

#endif  
