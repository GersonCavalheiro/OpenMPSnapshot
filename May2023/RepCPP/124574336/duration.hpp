





#ifndef BOOST_CHRONO_DURATION_HPP
#define BOOST_CHRONO_DURATION_HPP

#include <boost/chrono/config.hpp>
#include <boost/chrono/detail/static_assert.hpp>

#include <climits>
#include <limits>


#include <boost/mpl/logical.hpp>
#include <boost/ratio/ratio.hpp>
#include <boost/type_traits/common_type.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/chrono/detail/is_evenly_divisible_by.hpp>

#include <boost/cstdint.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/integer_traits.hpp>

#if !defined(BOOST_NO_CXX11_STATIC_ASSERT) || !defined(BOOST_CHRONO_USES_MPL_ASSERT)
#define BOOST_CHRONO_A_DURATION_REPRESENTATION_CAN_NOT_BE_A_DURATION        "A duration representation can not be a duration"
#define BOOST_CHRONO_SECOND_TEMPLATE_PARAMETER_OF_DURATION_MUST_BE_A_STD_RATIO "Second template parameter of duration must be a boost::ratio"
#define BOOST_CHRONO_DURATION_PERIOD_MUST_BE_POSITIVE "duration period must be positive"
#define BOOST_CHRONO_SECOND_TEMPLATE_PARAMETER_OF_TIME_POINT_MUST_BE_A_BOOST_CHRONO_DURATION "Second template parameter of time_point must be a boost::chrono::duration"
#endif

#ifndef BOOST_CHRONO_HEADER_ONLY
#include <boost/config/abi_prefix.hpp> 
#endif


namespace boost {
namespace chrono {

template <class Rep, class Period = ratio<1> >
class duration;

namespace detail
{
template <class T>
struct is_duration
: boost::false_type {};

template <class Rep, class Period>
struct is_duration<duration<Rep, Period> >
: boost::true_type  {};

template <class Duration, class Rep, bool = is_duration<Rep>::value>
struct duration_divide_result
{
};

template <class Duration, class Rep2,
bool = (
((boost::is_convertible<typename Duration::rep,
typename common_type<typename Duration::rep, Rep2>::type>::value))
&&  ((boost::is_convertible<Rep2,
typename common_type<typename Duration::rep, Rep2>::type>::value))
)
>
struct duration_divide_imp
{
};

template <class Rep1, class Period, class Rep2>
struct duration_divide_imp<duration<Rep1, Period>, Rep2, true>
{
typedef duration<typename common_type<Rep1, Rep2>::type, Period> type;
};

template <class Rep1, class Period, class Rep2>
struct duration_divide_result<duration<Rep1, Period>, Rep2, false>
: duration_divide_imp<duration<Rep1, Period>, Rep2>
{
};

template <class Rep, class Duration, bool = is_duration<Rep>::value>
struct duration_divide_result2
{
};

template <class Rep, class Duration,
bool = (
((boost::is_convertible<typename Duration::rep,
typename common_type<typename Duration::rep, Rep>::type>::value))
&&  ((boost::is_convertible<Rep,
typename common_type<typename Duration::rep, Rep>::type>::value))
)
>
struct duration_divide_imp2
{
};

template <class Rep1, class Rep2, class Period >
struct duration_divide_imp2<Rep1, duration<Rep2, Period>, true>
{
typedef double type;
};

template <class Rep1, class Rep2, class Period >
struct duration_divide_result2<Rep1, duration<Rep2, Period>, false>
: duration_divide_imp2<Rep1, duration<Rep2, Period> >
{
};

template <class Duration, class Rep, bool = is_duration<Rep>::value>
struct duration_modulo_result
{
};

template <class Duration, class Rep2,
bool = (
boost::is_convertible<Rep2,
typename common_type<typename Duration::rep, Rep2>::type>::value
)
>
struct duration_modulo_imp
{
};

template <class Rep1, class Period, class Rep2>
struct duration_modulo_imp<duration<Rep1, Period>, Rep2, true>
{
typedef duration<typename common_type<Rep1, Rep2>::type, Period> type;
};

template <class Rep1, class Period, class Rep2>
struct duration_modulo_result<duration<Rep1, Period>, Rep2, false>
: duration_modulo_imp<duration<Rep1, Period>, Rep2>
{
};

} 
} 



template <class Rep1, class Period1, class Rep2, class Period2>
struct common_type<chrono::duration<Rep1, Period1>,
chrono::duration<Rep2, Period2> >;


namespace chrono {

template <class Rep> struct treat_as_floating_point;
template <class Rep> struct duration_values;

typedef duration<boost::int_least64_t, nano> nanoseconds;    
typedef duration<boost::int_least64_t, micro> microseconds;  
typedef duration<boost::int_least64_t, milli> milliseconds;  
typedef duration<boost::int_least64_t> seconds;              
typedef duration<boost::int_least32_t, ratio< 60> > minutes; 
typedef duration<boost::int_least32_t, ratio<3600> > hours;  


namespace detail
{


template <class FromDuration, class ToDuration,
class Period,
bool PeriodNumEq1,
bool PeriodDenEq1>
struct duration_cast_aux;

template <class FromDuration, class ToDuration, class Period>
struct duration_cast_aux<FromDuration, ToDuration, Period, true, true>
{
BOOST_CONSTEXPR ToDuration operator()(const FromDuration& fd) const
{
return ToDuration(static_cast<typename ToDuration::rep>(fd.count()));
}
};

template <class FromDuration, class ToDuration, class Period>
struct duration_cast_aux<FromDuration, ToDuration, Period, true, false>
{
BOOST_CONSTEXPR ToDuration operator()(const FromDuration& fd) const
{
typedef typename common_type<
typename ToDuration::rep,
typename FromDuration::rep,
boost::intmax_t>::type C;
return ToDuration(static_cast<typename ToDuration::rep>(
static_cast<C>(fd.count()) / static_cast<C>(Period::den)));
}
};

template <class FromDuration, class ToDuration, class Period>
struct duration_cast_aux<FromDuration, ToDuration, Period, false, true>
{
BOOST_CONSTEXPR ToDuration operator()(const FromDuration& fd) const
{
typedef typename common_type<
typename ToDuration::rep,
typename FromDuration::rep,
boost::intmax_t>::type C;
return ToDuration(static_cast<typename ToDuration::rep>(
static_cast<C>(fd.count()) * static_cast<C>(Period::num)));
}
};

template <class FromDuration, class ToDuration, class Period>
struct duration_cast_aux<FromDuration, ToDuration, Period, false, false>
{
BOOST_CONSTEXPR ToDuration operator()(const FromDuration& fd) const
{
typedef typename common_type<
typename ToDuration::rep,
typename FromDuration::rep,
boost::intmax_t>::type C;
return ToDuration(static_cast<typename ToDuration::rep>(
static_cast<C>(fd.count()) * static_cast<C>(Period::num)
/ static_cast<C>(Period::den)));
}
};

template <class FromDuration, class ToDuration>
struct duration_cast {
typedef typename ratio_divide<typename FromDuration::period,
typename ToDuration::period>::type Period;
typedef duration_cast_aux<
FromDuration,
ToDuration,
Period,
Period::num == 1,
Period::den == 1
> Aux;
BOOST_CONSTEXPR ToDuration operator()(const FromDuration& fd) const
{
return Aux()(fd);
}
};

} 


template <class Rep>
struct treat_as_floating_point : boost::is_floating_point<Rep> {};


namespace detail {
template <class T, bool = is_arithmetic<T>::value>
struct chrono_numeric_limits {
static BOOST_CHRONO_LIB_CONSTEXPR T lowest() BOOST_CHRONO_LIB_NOEXCEPT_OR_THROW {return (std::numeric_limits<T>::min)  ();}
};

template <class T>
struct chrono_numeric_limits<T,true> {
static BOOST_CHRONO_LIB_CONSTEXPR T lowest() BOOST_CHRONO_LIB_NOEXCEPT_OR_THROW {return (std::numeric_limits<T>::min)  ();}
};

template <>
struct chrono_numeric_limits<float,true> {
static BOOST_CHRONO_LIB_CONSTEXPR float lowest() BOOST_CHRONO_LIB_NOEXCEPT_OR_THROW
{
return -(std::numeric_limits<float>::max) ();
}
};

template <>
struct chrono_numeric_limits<double,true> {
static BOOST_CHRONO_LIB_CONSTEXPR double lowest() BOOST_CHRONO_LIB_NOEXCEPT_OR_THROW
{
return -(std::numeric_limits<double>::max) ();
}
};

template <>
struct chrono_numeric_limits<long double,true> {
static BOOST_CHRONO_LIB_CONSTEXPR long double lowest() BOOST_CHRONO_LIB_NOEXCEPT_OR_THROW
{
return -(std::numeric_limits<long double>::max)();
}
};

template <class T>
struct numeric_limits : chrono_numeric_limits<typename remove_cv<T>::type>
{};

}
template <class Rep>
struct duration_values
{
static BOOST_CONSTEXPR Rep zero() {return Rep(0);}
static BOOST_CHRONO_LIB_CONSTEXPR Rep max BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
return (std::numeric_limits<Rep>::max)();
}

static BOOST_CHRONO_LIB_CONSTEXPR Rep min BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
return detail::numeric_limits<Rep>::lowest();
}
};

}  


template <class Rep1, class Period1, class Rep2, class Period2>
struct common_type<chrono::duration<Rep1, Period1>,
chrono::duration<Rep2, Period2> >
{
typedef chrono::duration<typename common_type<Rep1, Rep2>::type,
typename boost::ratio_gcd<Period1, Period2>::type> type;
};




namespace chrono {

template <class Rep, class Period>
class BOOST_SYMBOL_VISIBLE duration
{
BOOST_CHRONO_STATIC_ASSERT(!boost::chrono::detail::is_duration<Rep>::value,
BOOST_CHRONO_A_DURATION_REPRESENTATION_CAN_NOT_BE_A_DURATION, ());
BOOST_CHRONO_STATIC_ASSERT(boost::ratio_detail::is_ratio<typename Period::type>::value,
BOOST_CHRONO_SECOND_TEMPLATE_PARAMETER_OF_DURATION_MUST_BE_A_STD_RATIO, ());
BOOST_CHRONO_STATIC_ASSERT(Period::num>0,
BOOST_CHRONO_DURATION_PERIOD_MUST_BE_POSITIVE, ());
public:
typedef Rep rep;
typedef Period period;
private:
rep rep_;
public:

#if  defined   BOOST_CHRONO_DURATION_DEFAULTS_TO_ZERO
BOOST_FORCEINLINE BOOST_CONSTEXPR
duration() : rep_(duration_values<rep>::zero()) { }
#elif  defined   BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
BOOST_CONSTEXPR duration() {}
#else
BOOST_CONSTEXPR duration()  = default;
#endif
template <class Rep2>
BOOST_SYMBOL_VISIBLE BOOST_FORCEINLINE BOOST_CONSTEXPR
explicit duration(const Rep2& r
, typename boost::enable_if <
mpl::and_ <
boost::is_convertible<Rep2, rep>,
mpl::or_ <
treat_as_floating_point<rep>,
mpl::and_ <
mpl::not_ < treat_as_floating_point<rep> >,
mpl::not_ < treat_as_floating_point<Rep2> >
>
>
>
>::type* = 0
) : rep_(r) { }
#if  defined   BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
duration& operator=(const duration& rhs)
{
if (&rhs != this) rep_= rhs.rep_;
return *this;
}
#else
duration& operator=(const duration& rhs) = default;
#endif
template <class Rep2, class Period2>
BOOST_FORCEINLINE BOOST_CONSTEXPR
duration(const duration<Rep2, Period2>& d
, typename boost::enable_if <
mpl::or_ <
treat_as_floating_point<rep>,
mpl::and_ <
chrono_detail::is_evenly_divisible_by<Period2, period>,
mpl::not_ < treat_as_floating_point<Rep2> >
>
>
>::type* = 0
)
: rep_(chrono::detail::duration_cast<duration<Rep2, Period2>, duration>()(d).count()) {}


BOOST_CONSTEXPR
rep count() const {return rep_;}


BOOST_CONSTEXPR
duration  operator+() const {return duration(rep_);;}
BOOST_CONSTEXPR
duration  operator-() const {return duration(-rep_);}
duration& operator++()      {++rep_; return *this;}
duration  operator++(int)   {return duration(rep_++);}
duration& operator--()      {--rep_; return *this;}
duration  operator--(int)   {return duration(rep_--);}

duration& operator+=(const duration& d)
{
rep_ += d.count(); return *this;
}
duration& operator-=(const duration& d)
{
rep_ -= d.count(); return *this;
}

duration& operator*=(const rep& rhs) {rep_ *= rhs; return *this;}
duration& operator/=(const rep& rhs) {rep_ /= rhs; return *this;}
duration& operator%=(const rep& rhs) {rep_ %= rhs; return *this;}
duration& operator%=(const duration& rhs)
{
rep_ %= rhs.count(); return *this;
}

static BOOST_CONSTEXPR duration zero()
{
return duration(duration_values<rep>::zero());
}
static BOOST_CHRONO_LIB_CONSTEXPR duration min BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
return duration((duration_values<rep>::min)());
}
static BOOST_CHRONO_LIB_CONSTEXPR duration max BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
return duration((duration_values<rep>::max)());
}
};



template <class Rep1, class Period1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
typename common_type<duration<Rep1, Period1>, duration<Rep2, Period2> >::type
operator+(const duration<Rep1, Period1>& lhs,
const duration<Rep2, Period2>& rhs)
{
typedef typename common_type<duration<Rep1, Period1>,
duration<Rep2, Period2> >::type common_duration;
return common_duration(common_duration(lhs).count()+common_duration(rhs).count());
}


template <class Rep1, class Period1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
typename common_type<duration<Rep1, Period1>, duration<Rep2, Period2> >::type
operator-(const duration<Rep1, Period1>& lhs,
const duration<Rep2, Period2>& rhs)
{
typedef typename common_type<duration<Rep1, Period1>,
duration<Rep2, Period2> >::type common_duration;
return common_duration(common_duration(lhs).count()-common_duration(rhs).count());
}


template <class Rep1, class Period, class Rep2>
inline BOOST_CONSTEXPR
typename boost::enable_if <
mpl::and_ <
boost::is_convertible<Rep1, typename common_type<Rep1, Rep2>::type>,
boost::is_convertible<Rep2, typename common_type<Rep1, Rep2>::type>
>,
duration<typename common_type<Rep1, Rep2>::type, Period>
>::type
operator*(const duration<Rep1, Period>& d, const Rep2& s)
{
typedef typename common_type<Rep1, Rep2>::type common_rep;
typedef duration<common_rep, Period> common_duration;
return common_duration(common_duration(d).count()*static_cast<common_rep>(s));
}

template <class Rep1, class Period, class Rep2>
inline BOOST_CONSTEXPR
typename boost::enable_if <
mpl::and_ <
boost::is_convertible<Rep1, typename common_type<Rep1, Rep2>::type>,
boost::is_convertible<Rep2, typename common_type<Rep1, Rep2>::type>
>,
duration<typename common_type<Rep1, Rep2>::type, Period>
>::type
operator*(const Rep1& s, const duration<Rep2, Period>& d)
{
return d * s;
}


template <class Rep1, class Period, class Rep2>
inline BOOST_CONSTEXPR
typename boost::disable_if <boost::chrono::detail::is_duration<Rep2>,
typename boost::chrono::detail::duration_divide_result<
duration<Rep1, Period>, Rep2>::type
>::type
operator/(const duration<Rep1, Period>& d, const Rep2& s)
{
typedef typename common_type<Rep1, Rep2>::type common_rep;
typedef duration<common_rep, Period> common_duration;
return common_duration(common_duration(d).count()/static_cast<common_rep>(s));
}

template <class Rep1, class Period1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
typename common_type<Rep1, Rep2>::type
operator/(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs)
{
typedef typename common_type<duration<Rep1, Period1>,
duration<Rep2, Period2> >::type common_duration;
return common_duration(lhs).count() / common_duration(rhs).count();
}

#ifdef BOOST_CHRONO_EXTENSIONS
template <class Rep1, class Rep2, class Period>
inline BOOST_CONSTEXPR
typename boost::disable_if <boost::chrono::detail::is_duration<Rep1>,
typename boost::chrono::detail::duration_divide_result2<
Rep1, duration<Rep2, Period> >::type
>::type
operator/(const Rep1& s, const duration<Rep2, Period>& d)
{
typedef typename common_type<Rep1, Rep2>::type common_rep;
typedef duration<common_rep, Period> common_duration;
return static_cast<common_rep>(s)/common_duration(d).count();
}
#endif

template <class Rep1, class Period, class Rep2>
inline BOOST_CONSTEXPR
typename boost::disable_if <boost::chrono::detail::is_duration<Rep2>,
typename boost::chrono::detail::duration_modulo_result<
duration<Rep1, Period>, Rep2>::type
>::type
operator%(const duration<Rep1, Period>& d, const Rep2& s)
{
typedef typename common_type<Rep1, Rep2>::type common_rep;
typedef duration<common_rep, Period> common_duration;
return common_duration(common_duration(d).count()%static_cast<common_rep>(s));
}

template <class Rep1, class Period1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
typename common_type<duration<Rep1, Period1>, duration<Rep2, Period2> >::type
operator%(const duration<Rep1, Period1>& lhs,
const duration<Rep2, Period2>& rhs) {
typedef typename common_type<duration<Rep1, Period1>,
duration<Rep2, Period2> >::type common_duration;

return common_duration(common_duration(lhs).count()%common_duration(rhs).count());
}



namespace detail
{
template <class LhsDuration, class RhsDuration>
struct duration_eq
{
BOOST_CONSTEXPR bool operator()(const LhsDuration& lhs, const RhsDuration& rhs) const
{
typedef typename common_type<LhsDuration, RhsDuration>::type common_duration;
return common_duration(lhs).count() == common_duration(rhs).count();
}
};

template <class LhsDuration>
struct duration_eq<LhsDuration, LhsDuration>
{
BOOST_CONSTEXPR bool operator()(const LhsDuration& lhs, const LhsDuration& rhs) const
{
return lhs.count() == rhs.count();
}
};

template <class LhsDuration, class RhsDuration>
struct duration_lt
{
BOOST_CONSTEXPR bool operator()(const LhsDuration& lhs, const RhsDuration& rhs) const
{
typedef typename common_type<LhsDuration, RhsDuration>::type common_duration;
return common_duration(lhs).count() < common_duration(rhs).count();
}
};

template <class LhsDuration>
struct duration_lt<LhsDuration, LhsDuration>
{
BOOST_CONSTEXPR bool operator()(const LhsDuration& lhs, const LhsDuration& rhs) const
{
return lhs.count() < rhs.count();
}
};

} 


template <class Rep1, class Period1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
bool
operator==(const duration<Rep1, Period1>& lhs,
const duration<Rep2, Period2>& rhs)
{
return boost::chrono::detail::duration_eq<
duration<Rep1, Period1>, duration<Rep2, Period2> >()(lhs, rhs);
}


template <class Rep1, class Period1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
bool
operator!=(const duration<Rep1, Period1>& lhs,
const duration<Rep2, Period2>& rhs)
{
return !(lhs == rhs);
}


template <class Rep1, class Period1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
bool
operator< (const duration<Rep1, Period1>& lhs,
const duration<Rep2, Period2>& rhs)
{
return boost::chrono::detail::duration_lt<
duration<Rep1, Period1>, duration<Rep2, Period2> >()(lhs, rhs);
}


template <class Rep1, class Period1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
bool
operator> (const duration<Rep1, Period1>& lhs,
const duration<Rep2, Period2>& rhs)
{
return rhs < lhs;
}


template <class Rep1, class Period1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
bool
operator<=(const duration<Rep1, Period1>& lhs,
const duration<Rep2, Period2>& rhs)
{
return !(rhs < lhs);
}


template <class Rep1, class Period1, class Rep2, class Period2>
inline BOOST_CONSTEXPR
bool
operator>=(const duration<Rep1, Period1>& lhs,
const duration<Rep2, Period2>& rhs)
{
return !(lhs < rhs);
}


template <class ToDuration, class Rep, class Period>
inline BOOST_CONSTEXPR
typename boost::enable_if <
boost::chrono::detail::is_duration<ToDuration>, ToDuration>::type
duration_cast(const duration<Rep, Period>& fd)
{
return boost::chrono::detail::duration_cast<
duration<Rep, Period>, ToDuration>()(fd);
}

} 
} 

#ifndef BOOST_CHRONO_HEADER_ONLY
#include <boost/config/abi_suffix.hpp> 
#endif

#endif 
