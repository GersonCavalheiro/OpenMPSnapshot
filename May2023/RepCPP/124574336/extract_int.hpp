
#if !defined(BOOST_SPIRIT_X3_DETAIL_EXTRACT_INT_APRIL_17_2006_0816AM)
#define BOOST_SPIRIT_X3_DETAIL_EXTRACT_INT_APRIL_17_2006_0816AM

#include <boost/spirit/home/x3/support/unused.hpp>
#include <boost/spirit/home/x3/support/traits/attribute_type.hpp>
#include <boost/spirit/home/x3/support/traits/move_to.hpp>
#include <boost/spirit/home/x3/support/traits/numeric_traits.hpp>
#include <boost/spirit/home/support/char_encoding/ascii.hpp>

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/comparison/less.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/seq/elem.hpp>

#include <boost/utility/enable_if.hpp>

#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/make_unsigned.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/and.hpp>
#include <boost/limits.hpp>

#include <iterator> 

#if !defined(SPIRIT_NUMERICS_LOOP_UNROLL)
# define SPIRIT_NUMERICS_LOOP_UNROLL 3
#endif

namespace boost { namespace spirit { namespace x3 { namespace detail
{
template <typename T, unsigned Radix>
struct digits_traits;

template <int Digits, unsigned Radix>
struct digits2_to_n;

#define BOOST_SPIRIT_X3_LOG2 (#error)(#error)                                   \
(1000000)(1584960)(2000000)(2321920)(2584960)(2807350)                  \
(3000000)(3169920)(3321920)(3459430)(3584960)(3700430)                  \
(3807350)(3906890)(4000000)(4087460)(4169920)(4247920)                  \
(4321920)(4392310)(4459430)(4523560)(4584960)(4643850)                  \
(4700430)(4754880)(4807350)(4857980)(4906890)(4954190)                  \
(5000000)(5044390)(5087460)(5129280)(5169925)                           \


#define BOOST_PP_LOCAL_MACRO(Radix)                                             \
template <int Digits> struct digits2_to_n<Digits, Radix>                    \
{                                                                           \
BOOST_STATIC_CONSTANT(int, value = static_cast<int>(                    \
(Digits * 1000000) /                                                \
BOOST_PP_SEQ_ELEM(Radix, BOOST_SPIRIT_X3_LOG2)));               \
};                                                                          \


#define BOOST_PP_LOCAL_LIMITS (2, 36)
#include BOOST_PP_LOCAL_ITERATE()

#undef BOOST_SPIRIT_X3_LOG2

template <typename T, unsigned Radix>
struct digits_traits : digits2_to_n<std::numeric_limits<T>::digits, Radix>
{
static_assert(std::numeric_limits<T>::radix == 2, "");
};

template <typename T>
struct digits_traits<T, 10>
{
static int constexpr value = std::numeric_limits<T>::digits10;
};

template <unsigned Radix>
struct radix_traits
{
template <typename Char>
inline static bool is_valid(Char ch)
{
return (ch >= '0' && ch <= (Radix > 10 ? '9' : static_cast<Char>('0' + Radix -1)))
|| (Radix > 10 && ch >= 'a' && ch <= static_cast<Char>('a' + Radix -10 -1))
|| (Radix > 10 && ch >= 'A' && ch <= static_cast<Char>('A' + Radix -10 -1));
}

template <typename Char>
inline static unsigned digit(Char ch)
{
return (Radix <= 10 || (ch >= '0' && ch <= '9'))
? ch - '0'
: char_encoding::ascii::tolower(ch) - 'a' + 10;
}
};

template <unsigned Radix>
struct positive_accumulator
{
template <typename T, typename Char>
inline static void add(T& n, Char ch, mpl::false_) 
{
const int digit = radix_traits<Radix>::digit(ch);
n = n * T(Radix) + T(digit);
}

template <typename T, typename Char>
inline static bool add(T& n, Char ch, mpl::true_) 
{
T const max = (std::numeric_limits<T>::max)();
T const val = max / Radix;
if (n > val)
return false;

T tmp = n * Radix;

const int digit = radix_traits<Radix>::digit(ch);
if (tmp > max - digit)
return false;

n = tmp + static_cast<T>(digit);
return true;
}
};

template <unsigned Radix>
struct negative_accumulator
{
template <typename T, typename Char>
inline static void add(T& n, Char ch, mpl::false_) 
{
const int digit = radix_traits<Radix>::digit(ch);
n = n * T(Radix) - T(digit);
}

template <typename T, typename Char>
inline static bool add(T& n, Char ch, mpl::true_) 
{
T const min = (std::numeric_limits<T>::min)();
T const val = min / T(Radix);
if (n < val)
return false;

T tmp = n * Radix;

int const digit = radix_traits<Radix>::digit(ch);
if (tmp < min + digit)
return false;

n = tmp - static_cast<T>(digit);
return true;
}
};

template <unsigned Radix, typename Accumulator, int MaxDigits>
struct int_extractor
{
template <typename Char, typename T>
inline static bool
call(Char ch, std::size_t count, T& n, mpl::true_)
{
std::size_t constexpr
overflow_free = digits_traits<T, Radix>::value - 1;

if (count < overflow_free)
{
Accumulator::add(n, ch, mpl::false_());
}
else
{
if (!Accumulator::add(n, ch, mpl::true_()))
return false; 
}
return true;
}

template <typename Char, typename T>
inline static bool
call(Char ch, std::size_t , T& n, mpl::false_)
{
Accumulator::add(n, ch, mpl::false_());
return true;
}

template <typename Char>
inline static bool
call(Char , std::size_t , unused_type, mpl::false_)
{
return true;
}

template <typename Char, typename T>
inline static bool
call(Char ch, std::size_t count, T& n)
{
return call(ch, count, n
, mpl::bool_<
(   (MaxDigits < 0)
||  (MaxDigits > digits_traits<T, Radix>::value)
)
&& traits::check_overflow<T>::value
>()
);
}
};

template <int MaxDigits>
struct check_max_digits
{
inline static bool
call(std::size_t count)
{
return count < MaxDigits; 
}
};

template <>
struct check_max_digits<-1>
{
inline static bool
call(std::size_t )
{
return true; 
}
};

#define SPIRIT_NUMERIC_INNER_LOOP(z, x, data)                                   \
if (!check_max_digits<MaxDigits>::call(count + leading_zeros)           \
|| it == last)                                                      \
break;                                                              \
ch = *it;                                                               \
if (!radix_check::is_valid(ch) || !extractor::call(ch, count, val))     \
break;                                                              \
++it;                                                                   \
++count;                                                                \


template <
typename T, unsigned Radix, unsigned MinDigits, int MaxDigits
, typename Accumulator = positive_accumulator<Radix>
, bool Accumulate = false
>
struct extract_int
{
#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(push)
# pragma warning(disable: 4127)   
#endif
template <typename Iterator, typename Attribute>
inline static bool
parse_main(
Iterator& first
, Iterator const& last
, Attribute& attr)
{
typedef radix_traits<Radix> radix_check;
typedef int_extractor<Radix, Accumulator, MaxDigits> extractor;
typedef typename
std::iterator_traits<Iterator>::value_type
char_type;

Iterator it = first;
std::size_t leading_zeros = 0;
if (!Accumulate)
{
while (it != last && *it == '0' && leading_zeros < MaxDigits)
{
++it;
++leading_zeros;
}
}

typedef typename
traits::attribute_type<Attribute>::type
attribute_type;

attribute_type val = Accumulate ? attr : attribute_type(0);
std::size_t count = 0;
char_type ch;

while (true)
{
BOOST_PP_REPEAT(
SPIRIT_NUMERICS_LOOP_UNROLL
, SPIRIT_NUMERIC_INNER_LOOP, _)
}

if (count + leading_zeros >= MinDigits)
{
traits::move_to(val, attr);
first = it;
return true;
}
return false;
}
#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(pop)
#endif

template <typename Iterator>
inline static bool
parse(
Iterator& first
, Iterator const& last
, unused_type)
{
T n = 0; 
return parse_main(first, last, n);
}

template <typename Iterator, typename Attribute>
inline static bool
parse(
Iterator& first
, Iterator const& last
, Attribute& attr)
{
return parse_main(first, last, attr);
}
};
#undef SPIRIT_NUMERIC_INNER_LOOP

#define SPIRIT_NUMERIC_INNER_LOOP(z, x, data)                                   \
if (it == last)                                                         \
break;                                                              \
ch = *it;                                                               \
if (!radix_check::is_valid(ch))                                         \
break;                                                              \
if (!extractor::call(ch, count, val))                                   \
return false;                                                       \
++it;                                                                   \
++count;                                                                \


template <typename T, unsigned Radix, typename Accumulator, bool Accumulate>
struct extract_int<T, Radix, 1, -1, Accumulator, Accumulate>
{
#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(push)
# pragma warning(disable: 4127)   
#endif
template <typename Iterator, typename Attribute>
inline static bool
parse_main(
Iterator& first
, Iterator const& last
, Attribute& attr)
{
typedef radix_traits<Radix> radix_check;
typedef int_extractor<Radix, Accumulator, -1> extractor;
typedef typename
std::iterator_traits<Iterator>::value_type
char_type;

Iterator it = first;
std::size_t count = 0;
if (!Accumulate)
{
while (it != last && *it == '0')
{
++it;
++count;
}

if (it == last)
{
if (count == 0) 
return false;
attr = 0;
first = it;
return true;
}
}

typedef typename
traits::attribute_type<Attribute>::type
attribute_type;

attribute_type val = Accumulate ? attr : attribute_type(0);
char_type ch = *it;

if (!radix_check::is_valid(ch) || !extractor::call(ch, 0, val))
{
if (count == 0) 
return false;
traits::move_to(val, attr);
first = it;
return true;
}

count = 0;
++it;
while (true)
{
BOOST_PP_REPEAT(
SPIRIT_NUMERICS_LOOP_UNROLL
, SPIRIT_NUMERIC_INNER_LOOP, _)
}

traits::move_to(val, attr);
first = it;
return true;
}
#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(pop)
#endif

template <typename Iterator>
inline static bool
parse(
Iterator& first
, Iterator const& last
, unused_type)
{
T n = 0; 
return parse_main(first, last, n);
}

template <typename Iterator, typename Attribute>
inline static bool
parse(
Iterator& first
, Iterator const& last
, Attribute& attr)
{
return parse_main(first, last, attr);
}
};

#undef SPIRIT_NUMERIC_INNER_LOOP
}}}}

#endif
