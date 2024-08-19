





#ifndef BOOST_RATIO_RATIO_FWD_HPP
#define BOOST_RATIO_RATIO_FWD_HPP

#include <boost/ratio/config.hpp>

#if defined(__GNUC__) && (__GNUC__ >= 4)
#pragma GCC system_header
#endif

namespace boost
{


template <boost::intmax_t N, boost::intmax_t D = 1> class ratio;

template <class R1, class R2> struct ratio_add;
template <class R1, class R2> struct ratio_subtract;
template <class R1, class R2> struct ratio_multiply;
template <class R1, class R2> struct ratio_divide;
#ifdef BOOST_RATIO_EXTENSIONS
template <class R1, class R2> struct ratio_gcd;
template <class R1, class R2> struct ratio_lcm;
template <class R> struct ratio_negate;
template <class R> struct ratio_abs;
template <class R> struct ratio_sign;
template <class R, int P> struct ratio_power;
#endif

template <class R1, class R2> struct ratio_equal;
template <class R1, class R2> struct ratio_not_equal;
template <class R1, class R2> struct ratio_less;
template <class R1, class R2> struct ratio_less_equal;
template <class R1, class R2> struct ratio_greater;
template <class R1, class R2> struct ratio_greater_equal;

typedef ratio<BOOST_RATIO_INTMAX_C(1), BOOST_RATIO_INTMAX_C(1000000000000000000)> atto;
typedef ratio<BOOST_RATIO_INTMAX_C(1),    BOOST_RATIO_INTMAX_C(1000000000000000)> femto;
typedef ratio<BOOST_RATIO_INTMAX_C(1),       BOOST_RATIO_INTMAX_C(1000000000000)> pico;
typedef ratio<BOOST_RATIO_INTMAX_C(1),          BOOST_RATIO_INTMAX_C(1000000000)> nano;
typedef ratio<BOOST_RATIO_INTMAX_C(1),             BOOST_RATIO_INTMAX_C(1000000)> micro;
typedef ratio<BOOST_RATIO_INTMAX_C(1),                BOOST_RATIO_INTMAX_C(1000)> milli;
typedef ratio<BOOST_RATIO_INTMAX_C(1),                 BOOST_RATIO_INTMAX_C(100)> centi;
typedef ratio<BOOST_RATIO_INTMAX_C(1),                  BOOST_RATIO_INTMAX_C(10)> deci;
typedef ratio<                 BOOST_RATIO_INTMAX_C(10), BOOST_RATIO_INTMAX_C(1)> deca;
typedef ratio<                BOOST_RATIO_INTMAX_C(100), BOOST_RATIO_INTMAX_C(1)> hecto;
typedef ratio<               BOOST_RATIO_INTMAX_C(1000), BOOST_RATIO_INTMAX_C(1)> kilo;
typedef ratio<            BOOST_RATIO_INTMAX_C(1000000), BOOST_RATIO_INTMAX_C(1)> mega;
typedef ratio<         BOOST_RATIO_INTMAX_C(1000000000), BOOST_RATIO_INTMAX_C(1)> giga;
typedef ratio<      BOOST_RATIO_INTMAX_C(1000000000000), BOOST_RATIO_INTMAX_C(1)> tera;
typedef ratio<   BOOST_RATIO_INTMAX_C(1000000000000000), BOOST_RATIO_INTMAX_C(1)> peta;
typedef ratio<BOOST_RATIO_INTMAX_C(1000000000000000000), BOOST_RATIO_INTMAX_C(1)> exa;

#ifdef BOOST_RATIO_EXTENSIONS

#define BOOST_RATIO_1024 BOOST_RATIO_INTMAX_C(1024)

typedef ratio<                                                                                     BOOST_RATIO_1024> kibi;
typedef ratio<                                                                    BOOST_RATIO_1024*BOOST_RATIO_1024> mebi;
typedef ratio<                                                   BOOST_RATIO_1024*BOOST_RATIO_1024*BOOST_RATIO_1024> gibi;
typedef ratio<                                  BOOST_RATIO_1024*BOOST_RATIO_1024*BOOST_RATIO_1024*BOOST_RATIO_1024> tebi;
typedef ratio<                 BOOST_RATIO_1024*BOOST_RATIO_1024*BOOST_RATIO_1024*BOOST_RATIO_1024*BOOST_RATIO_1024> pebi;
typedef ratio<BOOST_RATIO_1024*BOOST_RATIO_1024*BOOST_RATIO_1024*BOOST_RATIO_1024*BOOST_RATIO_1024*BOOST_RATIO_1024> exbi;

#endif
}  


#endif  
