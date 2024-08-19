#ifndef BOOST_BASIC_TIMED_MUTEX_WIN32_HPP
#define BOOST_BASIC_TIMED_MUTEX_WIN32_HPP


#include <boost/assert.hpp>
#include <boost/thread/win32/thread_primitives.hpp>
#include <boost/thread/win32/interlocked_read.hpp>
#include <boost/thread/thread_time.hpp>
#if defined BOOST_THREAD_USES_DATETIME
#include <boost/thread/xtime.hpp>
#endif
#include <boost/detail/interlocked.hpp>
#ifdef BOOST_THREAD_USES_CHRONO
#include <boost/chrono/system_clocks.hpp>
#include <boost/chrono/ceil.hpp>
#endif
#include <boost/thread/detail/platform_time.hpp>

#include <boost/config/abi_prefix.hpp>

namespace boost
{
namespace detail
{
struct basic_timed_mutex
{
BOOST_STATIC_CONSTANT(unsigned char,lock_flag_bit=31);
BOOST_STATIC_CONSTANT(unsigned char,event_set_flag_bit=30);
BOOST_STATIC_CONSTANT(long,lock_flag_value=1<<lock_flag_bit);
BOOST_STATIC_CONSTANT(long,event_set_flag_value=1<<event_set_flag_bit);
long active_count;
void* event;

void initialize()
{
active_count=0;
event=0;
}

void destroy()
{
#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4312)
#endif
void* const old_event=BOOST_INTERLOCKED_EXCHANGE_POINTER(&event,0);
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
if(old_event)
{
winapi::CloseHandle(old_event);
}
}

bool try_lock() BOOST_NOEXCEPT
{
return !win32::interlocked_bit_test_and_set(&active_count,lock_flag_bit);
}

void lock()
{
if(try_lock())
{
return;
}
long old_count=active_count;
mark_waiting_and_try_lock(old_count);

if(old_count&lock_flag_value)
{
void* const sem=get_event();

do
{
if(winapi::WaitForSingleObjectEx(sem,::boost::detail::win32::infinite,0)==0)
{
clear_waiting_and_try_lock(old_count);
}
}
while(old_count&lock_flag_value);
}
}

void mark_waiting_and_try_lock(long& old_count)
{
for(;;)
{
bool const was_locked=(old_count&lock_flag_value) ? true : false;
long const new_count=was_locked?(old_count+1):(old_count|lock_flag_value);
long const current=BOOST_INTERLOCKED_COMPARE_EXCHANGE(&active_count,new_count,old_count);
if(current==old_count)
{
if(was_locked)
old_count=new_count;
break;
}
old_count=current;
}
}

void clear_waiting_and_try_lock(long& old_count)
{
old_count&=~lock_flag_value;
old_count|=event_set_flag_value;
for(;;)
{
long const new_count=((old_count&lock_flag_value)?old_count:((old_count-1)|lock_flag_value))&~event_set_flag_value;
long const current=BOOST_INTERLOCKED_COMPARE_EXCHANGE(&active_count,new_count,old_count);
if(current==old_count)
{
break;
}
old_count=current;
}
}

private:
unsigned long getMs(detail::platform_duration const& d)
{
return static_cast<unsigned long>(d.getMs());
}

template <typename Duration>
unsigned long getMs(Duration const& d)
{
return static_cast<unsigned long>(chrono::ceil<chrono::milliseconds>(d).count());
}

template <typename Clock, typename Timepoint, typename Duration>
bool do_lock_until(Timepoint const& t, Duration const& max)
{
if(try_lock())
{
return true;
}

long old_count=active_count;
mark_waiting_and_try_lock(old_count);

if(old_count&lock_flag_value)
{
void* const sem=get_event();

do
{
Duration d(t - Clock::now());
if(d <= Duration::zero()) 
{
BOOST_INTERLOCKED_DECREMENT(&active_count);
return false;
}
if(max != Duration::zero())
{
d = (std::min)(d, max);
}
if(winapi::WaitForSingleObjectEx(sem,getMs(d),0)==0)
{
clear_waiting_and_try_lock(old_count);
}
}
while(old_count&lock_flag_value);
}
return true;
}
public:

#if defined BOOST_THREAD_USES_DATETIME
bool timed_lock(::boost::system_time const& wait_until)
{
const detail::real_platform_timepoint t(wait_until);
return do_lock_until<detail::real_platform_clock>(t, detail::platform_milliseconds(BOOST_THREAD_POLL_INTERVAL_MILLISECONDS));
}

template<typename Duration>
bool timed_lock(Duration const& timeout)
{
const detail::mono_platform_timepoint t(detail::mono_platform_clock::now() + detail::platform_duration(timeout));
return do_lock_until<detail::mono_platform_clock>(t, detail::platform_duration::zero());
}

bool timed_lock(boost::xtime const& timeout)
{
return timed_lock(boost::system_time(timeout));
}
#endif
#ifdef BOOST_THREAD_USES_CHRONO
template <class Rep, class Period>
bool try_lock_for(const chrono::duration<Rep, Period>& rel_time)
{
const chrono::steady_clock::time_point t(chrono::steady_clock::now() + rel_time);
typedef typename chrono::duration<Rep, Period> Duration;
typedef typename common_type<Duration, typename chrono::steady_clock::duration>::type common_duration;
return do_lock_until<chrono::steady_clock>(t, common_duration::zero());
}
template <class Duration>
bool try_lock_until(const chrono::time_point<chrono::steady_clock, Duration>& t)
{
typedef typename common_type<Duration, typename chrono::steady_clock::duration>::type common_duration;
return do_lock_until<chrono::steady_clock>(t, common_duration::zero());
}
template <class Clock, class Duration>
bool try_lock_until(const chrono::time_point<Clock, Duration>& t)
{
typedef typename common_type<Duration, typename Clock::duration>::type common_duration;
return do_lock_until<Clock>(t, common_duration(chrono::milliseconds(BOOST_THREAD_POLL_INTERVAL_MILLISECONDS)));
}
#endif

void unlock()
{
long const old_count=BOOST_INTERLOCKED_EXCHANGE_ADD(&active_count,lock_flag_value);
if(!(old_count&event_set_flag_value) && (old_count>lock_flag_value))
{
if(!win32::interlocked_bit_test_and_set(&active_count,event_set_flag_bit))
{
winapi::SetEvent(get_event());
}
}
}

private:
void* get_event()
{
void* current_event=::boost::detail::interlocked_read_acquire(&event);

if(!current_event)
{
void* const new_event=win32::create_anonymous_event(win32::auto_reset_event,win32::event_initially_reset);
#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4311)
#pragma warning(disable:4312)
#endif
void* const old_event=BOOST_INTERLOCKED_COMPARE_EXCHANGE_POINTER(&event,new_event,0);
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
if(old_event!=0)
{
winapi::CloseHandle(new_event);
return old_event;
}
else
{
return new_event;
}
}
return current_event;
}

};

}
}

#define BOOST_BASIC_TIMED_MUTEX_INITIALIZER {0}

#include <boost/config/abi_suffix.hpp>

#endif
