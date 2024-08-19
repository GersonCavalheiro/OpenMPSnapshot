#ifndef BOOST_INTERPROCESS_DETAIL_SP_COUNTED_BASE_ATOMIC_HPP_INCLUDED
#define BOOST_INTERPROCESS_DETAIL_SP_COUNTED_BASE_ATOMIC_HPP_INCLUDED

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
# pragma once
#endif


#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/detail/atomic.hpp>
#include <typeinfo>

namespace boost {

namespace interprocess {

namespace ipcdetail {

class sp_counted_base
{
private:

sp_counted_base( sp_counted_base const & );
sp_counted_base & operator= ( sp_counted_base const & );

boost::uint32_t use_count_;        
boost::uint32_t weak_count_;       

public:

sp_counted_base(): use_count_( 1 ), weak_count_( 1 )
{}

~sp_counted_base() 
{}

void add_ref_copy()
{
ipcdetail::atomic_inc32( &use_count_ );
}

bool add_ref_lock() 
{
for( ;; )
{
boost::uint32_t tmp = static_cast< boost::uint32_t const volatile& >( use_count_ );
if( tmp == 0 ) return false;
if( ipcdetail::atomic_cas32( &use_count_, tmp + 1, tmp ) == tmp )
return true;
}
}

bool ref_release() 
{ return 1 == ipcdetail::atomic_dec32( &use_count_ );  }

void weak_add_ref() 
{ ipcdetail::atomic_inc32( &weak_count_ ); }

bool weak_release() 
{ return 1 == ipcdetail::atomic_dec32( &weak_count_ ); }

long use_count() const 
{ return (long)static_cast<boost::uint32_t const volatile &>( use_count_ ); }
};

} 

} 

} 

#include <boost/interprocess/detail/config_end.hpp>

#endif  
