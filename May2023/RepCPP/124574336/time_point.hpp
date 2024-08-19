





#ifndef BOOST_CHRONO_TIME_POINT_HPP
#define BOOST_CHRONO_TIME_POINT_HPP

#include <boost/chrono/duration.hpp>

#ifndef BOOST_CHRONO_HEADER_ONLY
#include <boost/config/abi_prefix.hpp> 
#endif


namespace boost {
namespace chrono {

template <class Clock, class Duration = typename Clock::duration>
class time_point;


} 



template <class Clock, class Duration1, class Duration2>
struct common_type<chrono::time_point<Clock, Duration1>,
chrono::time_point<Clock, Duration2> >;




template <class Clock, class Duration1, class Duration2>
struct common_type<chrono::time_point<Clock, Duration1>,
chrono::time_point<Clock, Duration2> >
{
typedef chrono::time_point<Clock,
typename common_type<Duration1, Duration2>::type> type;
};



namespace chrono {

template <class Clock, class Duration1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
time_point<Clock,
typename common_type<Duration1, duration<Rep2, Period2> >::type>
operator+(
const time_point<Clock, Duration1>& lhs,
const duration<Rep2, Period2>& rhs);
template <class Rep1, class Period1, class Clock, class Duration2>
inline BOOST_CONSTEXPR
time_point<Clock,
typename common_type<duration<Rep1, Period1>, Duration2>::type>
operator+(
const duration<Rep1, Period1>& lhs,
const time_point<Clock, Duration2>& rhs);
template <class Clock, class Duration1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
time_point<Clock,
typename common_type<Duration1, duration<Rep2, Period2> >::type>
operator-(
const time_point<Clock, Duration1>& lhs,
const duration<Rep2, Period2>& rhs);
template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
typename common_type<Duration1, Duration2>::type
operator-(
const time_point<Clock, Duration1>& lhs,
const time_point<Clock,
Duration2>& rhs);

template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool operator==(
const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs);
template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool operator!=(
const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs);
template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool operator< (
const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs);
template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool operator<=(
const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs);
template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool operator> (
const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs);
template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool operator>=(
const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs);

template <class ToDuration, class Clock, class Duration>
inline BOOST_CONSTEXPR
time_point<Clock, ToDuration> time_point_cast(const time_point<Clock, Duration>& t);


template <class Clock, class Duration>
class time_point
{
BOOST_CHRONO_STATIC_ASSERT(boost::chrono::detail::is_duration<Duration>::value,
BOOST_CHRONO_SECOND_TEMPLATE_PARAMETER_OF_TIME_POINT_MUST_BE_A_BOOST_CHRONO_DURATION, (Duration));
public:
typedef Clock                     clock;
typedef Duration                  duration;
typedef typename duration::rep    rep;
typedef typename duration::period period;
typedef Duration                  difference_type;

private:
duration d_;

public:
BOOST_FORCEINLINE BOOST_CONSTEXPR
time_point() : d_(duration::zero())
{}
BOOST_FORCEINLINE BOOST_CONSTEXPR
explicit time_point(const duration& d)
: d_(d)
{}

template <class Duration2>
BOOST_FORCEINLINE BOOST_CONSTEXPR
time_point(const time_point<clock, Duration2>& t
, typename boost::enable_if
<
boost::is_convertible<Duration2, duration>
>::type* = 0
)
: d_(t.time_since_epoch())
{
}

BOOST_CONSTEXPR
duration time_since_epoch() const
{
return d_;
}


#ifdef BOOST_CHRONO_EXTENSIONS
BOOST_CONSTEXPR
time_point  operator+() const {return *this;}
BOOST_CONSTEXPR
time_point  operator-() const {return time_point(-d_);}
time_point& operator++()      {++d_; return *this;}
time_point  operator++(int)   {return time_point(d_++);}
time_point& operator--()      {--d_; return *this;}
time_point  operator--(int)   {return time_point(d_--);}

time_point& operator+=(const rep& r) {d_ += duration(r); return *this;}
time_point& operator-=(const rep& r) {d_ -= duration(r); return *this;}

#endif

time_point& operator+=(const duration& d) {d_ += d; return *this;}
time_point& operator-=(const duration& d) {d_ -= d; return *this;}


static BOOST_CHRONO_LIB_CONSTEXPR time_point
min BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
return time_point((duration::min)());
}
static BOOST_CHRONO_LIB_CONSTEXPR time_point
max BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
return time_point((duration::max)());
}
};



template <class Clock, class Duration1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
time_point<Clock,
typename common_type<Duration1, duration<Rep2, Period2> >::type>
operator+(const time_point<Clock, Duration1>& lhs,
const duration<Rep2, Period2>& rhs)
{
typedef typename common_type<Duration1, duration<Rep2, Period2> >::type CDuration;
typedef time_point<
Clock,
CDuration
> TimeResult;
return TimeResult(lhs.time_since_epoch() + CDuration(rhs));
}


template <class Rep1, class Period1, class Clock, class Duration2>
inline BOOST_CONSTEXPR
time_point<Clock,
typename common_type<duration<Rep1, Period1>, Duration2>::type>
operator+(const duration<Rep1, Period1>& lhs,
const time_point<Clock, Duration2>& rhs)
{
return rhs + lhs;
}


template <class Clock, class Duration1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
time_point<Clock,
typename common_type<Duration1, duration<Rep2, Period2> >::type>
operator-(const time_point<Clock, Duration1>& lhs,
const duration<Rep2, Period2>& rhs)
{
return lhs + (-rhs);
}


template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
typename common_type<Duration1, Duration2>::type
operator-(const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs)
{
return lhs.time_since_epoch() - rhs.time_since_epoch();
}



template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool
operator==(const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs)
{
return lhs.time_since_epoch() == rhs.time_since_epoch();
}


template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool
operator!=(const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs)
{
return !(lhs == rhs);
}


template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool
operator<(const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs)
{
return lhs.time_since_epoch() < rhs.time_since_epoch();
}


template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool
operator>(const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs)
{
return rhs < lhs;
}


template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool
operator<=(const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs)
{
return !(rhs < lhs);
}


template <class Clock, class Duration1, class Duration2>
inline BOOST_CONSTEXPR
bool
operator>=(const time_point<Clock, Duration1>& lhs,
const time_point<Clock, Duration2>& rhs)
{
return !(lhs < rhs);
}


template <class ToDuration, class Clock, class Duration>
inline BOOST_CONSTEXPR
time_point<Clock, ToDuration>
time_point_cast(const time_point<Clock, Duration>& t)
{
return time_point<Clock, ToDuration>(
duration_cast<ToDuration>(t.time_since_epoch()));
}

} 
} 

#ifndef BOOST_CHRONO_HEADER_ONLY
#include <boost/config/abi_suffix.hpp> 
#endif

#endif 
