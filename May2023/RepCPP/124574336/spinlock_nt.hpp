#ifndef BOOST_SMART_PTR_DETAIL_SPINLOCK_NT_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SPINLOCK_NT_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/assert.hpp>

#if defined(BOOST_SP_REPORT_IMPLEMENTATION)

#include <boost/config/pragma_message.hpp>
BOOST_PRAGMA_MESSAGE("Using single-threaded spinlock emulation")

#endif

namespace boost
{

namespace detail
{

class spinlock
{
public:

bool locked_;

public:

inline bool try_lock()
{
if( locked_ )
{
return false;
}
else
{
locked_ = true;
return true;
}
}

inline void lock()
{
BOOST_ASSERT( !locked_ );
locked_ = true;
}

inline void unlock()
{
BOOST_ASSERT( locked_ );
locked_ = false;
}

public:

class scoped_lock
{
private:

spinlock & sp_;

scoped_lock( scoped_lock const & );
scoped_lock & operator=( scoped_lock const & );

public:

explicit scoped_lock( spinlock & sp ): sp_( sp )
{
sp.lock();
}

~scoped_lock()
{
sp_.unlock();
}
};
};

} 
} 

#define BOOST_DETAIL_SPINLOCK_INIT { false }

#endif 
