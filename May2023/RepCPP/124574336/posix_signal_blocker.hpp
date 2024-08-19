
#ifndef BOOST_ASIO_DETAIL_POSIX_SIGNAL_BLOCKER_HPP
#define BOOST_ASIO_DETAIL_POSIX_SIGNAL_BLOCKER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_PTHREADS)

#include <csignal>
#include <pthread.h>
#include <signal.h>
#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class posix_signal_blocker
: private noncopyable
{
public:
posix_signal_blocker()
: blocked_(false)
{
sigset_t new_mask;
sigfillset(&new_mask);
blocked_ = (pthread_sigmask(SIG_BLOCK, &new_mask, &old_mask_) == 0);
}

~posix_signal_blocker()
{
if (blocked_)
pthread_sigmask(SIG_SETMASK, &old_mask_, 0);
}

void block()
{
if (!blocked_)
{
sigset_t new_mask;
sigfillset(&new_mask);
blocked_ = (pthread_sigmask(SIG_BLOCK, &new_mask, &old_mask_) == 0);
}
}

void unblock()
{
if (blocked_)
blocked_ = (pthread_sigmask(SIG_SETMASK, &old_mask_, 0) != 0);
}

private:
bool blocked_;

sigset_t old_mask_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
