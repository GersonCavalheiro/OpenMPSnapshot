




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011
#  include <type_traits>
#endif

namespace hydra_thrust
{

template<typename T> class device_reference;

namespace detail
{
template<typename T, T v>
struct integral_constant
{
HYDRA_THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT T value = v;

typedef T                       value_type;
typedef integral_constant<T, v> type;

#if HYDRA_THRUST_CPP_DIALECT >= 2011
constexpr integral_constant() = default;

constexpr integral_constant(integral_constant const&) = default;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
constexpr 
#endif
integral_constant& operator=(integral_constant const&) = default;

constexpr __host__ __device__
integral_constant(std::integral_constant<T, v>) {}
#endif

HYDRA_THRUST_CONSTEXPR __host__ __device__ operator value_type() const HYDRA_THRUST_NOEXCEPT { return value; }
HYDRA_THRUST_CONSTEXPR __host__ __device__ value_type operator()() const HYDRA_THRUST_NOEXCEPT { return value; }
};

typedef integral_constant<bool, true>  true_type;

typedef integral_constant<bool, false> false_type;

template<typename T> struct is_integral                           : public false_type {};
template<>           struct is_integral<bool>                     : public true_type {};
template<>           struct is_integral<char>                     : public true_type {};
template<>           struct is_integral<signed char>              : public true_type {};
template<>           struct is_integral<unsigned char>            : public true_type {};
template<>           struct is_integral<short>                    : public true_type {};
template<>           struct is_integral<unsigned short>           : public true_type {};
template<>           struct is_integral<int>                      : public true_type {};
template<>           struct is_integral<unsigned int>             : public true_type {};
template<>           struct is_integral<long>                     : public true_type {};
template<>           struct is_integral<unsigned long>            : public true_type {};
template<>           struct is_integral<long long>                : public true_type {};
template<>           struct is_integral<unsigned long long>       : public true_type {};
template<>           struct is_integral<const bool>               : public true_type {};
template<>           struct is_integral<const char>               : public true_type {};
template<>           struct is_integral<const unsigned char>      : public true_type {};
template<>           struct is_integral<const short>              : public true_type {};
template<>           struct is_integral<const unsigned short>     : public true_type {};
template<>           struct is_integral<const int>                : public true_type {};
template<>           struct is_integral<const unsigned int>       : public true_type {};
template<>           struct is_integral<const long>               : public true_type {};
template<>           struct is_integral<const unsigned long>      : public true_type {};
template<>           struct is_integral<const long long>          : public true_type {};
template<>           struct is_integral<const unsigned long long> : public true_type {};

template<typename T> struct is_floating_point              : public false_type {};
template<>           struct is_floating_point<float>       : public true_type {};
template<>           struct is_floating_point<double>      : public true_type {};
template<>           struct is_floating_point<long double> : public true_type {};

template<typename T> struct is_arithmetic               : public is_integral<T> {};
template<>           struct is_arithmetic<float>        : public true_type {};
template<>           struct is_arithmetic<double>       : public true_type {};
template<>           struct is_arithmetic<const float>  : public true_type {};
template<>           struct is_arithmetic<const double> : public true_type {};

template<typename T> struct is_pointer      : public false_type {};
template<typename T> struct is_pointer<T *> : public true_type  {};

template<typename T> struct is_device_ptr  : public false_type {};

template<typename T> struct is_void             : public false_type {};
template<>           struct is_void<void>       : public true_type {};
template<>           struct is_void<const void> : public true_type {};

template<typename T> struct is_non_bool_integral       : public is_integral<T> {};
template<>           struct is_non_bool_integral<bool> : public false_type {};

template<typename T> struct is_non_bool_arithmetic       : public is_arithmetic<T> {};
template<>           struct is_non_bool_arithmetic<bool> : public false_type {};

template<typename T> struct is_pod
: public integral_constant<
bool,
is_void<T>::value || is_pointer<T>::value || is_arithmetic<T>::value
#if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC || \
HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG
|| __is_pod(T)
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
|| __is_pod(T)
#endif 
#endif 
>
{};


template<typename T> struct has_trivial_constructor
: public integral_constant<
bool,
is_pod<T>::value
#if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC || \
HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG
|| __has_trivial_constructor(T)
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
|| __has_trivial_constructor(T)
#endif 
#endif 
>
{};

template<typename T> struct has_trivial_copy_constructor
: public integral_constant<
bool,
is_pod<T>::value
#if HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC || \
HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG
|| __has_trivial_copy(T)
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
|| __has_trivial_copy(T)
#endif 
#endif 
>
{};

template<typename T> struct has_trivial_destructor : public is_pod<T> {};

template<typename T> struct is_const          : public false_type {};
template<typename T> struct is_const<const T> : public true_type {};

template<typename T> struct is_volatile             : public false_type {};
template<typename T> struct is_volatile<volatile T> : public true_type {};

template<typename T>
struct add_const
{
typedef T const type;
}; 

template<typename T>
struct remove_const
{
typedef T type;
}; 

template<typename T>
struct remove_const<const T>
{
typedef T type;
}; 

template<typename T>
struct add_volatile
{
typedef volatile T type;
}; 

template<typename T>
struct remove_volatile
{
typedef T type;
}; 

template<typename T>
struct remove_volatile<volatile T>
{
typedef T type;
}; 

template<typename T>
struct add_cv
{
typedef const volatile T type;
}; 

template<typename T>
struct remove_cv
{
typedef typename remove_const<typename remove_volatile<T>::type>::type type;
}; 


template<typename T> struct is_reference     : public false_type {};
template<typename T> struct is_reference<T&> : public true_type {};

template<typename T> struct is_proxy_reference  : public false_type {};

template<typename T> struct is_device_reference                                : public false_type {};
template<typename T> struct is_device_reference< hydra_thrust::device_reference<T> > : public true_type {};


template<typename _Tp, bool = (is_void<_Tp>::value || is_reference<_Tp>::value)>
struct __add_reference_helper
{ typedef _Tp&    type; };

template<typename _Tp>
struct __add_reference_helper<_Tp, true>
{ typedef _Tp     type; };

template<typename _Tp>
struct add_reference
: public __add_reference_helper<_Tp>{};

template<typename T>
struct remove_reference
{
typedef T type;
}; 

template<typename T>
struct remove_reference<T&>
{
typedef T type;
}; 

template<typename T1, typename T2>
struct is_same
: public false_type
{
}; 

template<typename T>
struct is_same<T,T>
: public true_type
{
}; 

template<typename T1, typename T2>
struct lazy_is_same
: is_same<typename T1::type, typename T2::type>
{
}; 

template<typename T1, typename T2>
struct is_different
: public true_type
{
}; 

template<typename T>
struct is_different<T,T>
: public false_type
{
}; 

template<typename T1, typename T2>
struct lazy_is_different
: is_different<typename T1::type, typename T2::type>
{
}; 

#if HYDRA_THRUST_CPP_DIALECT >= 2011

using std::is_convertible;

#else

namespace tt_detail
{

template<typename T>
struct is_int_or_cref
{
typedef typename remove_reference<T>::type type_sans_ref;
static const bool value = (is_integral<T>::value
|| (is_integral<type_sans_ref>::value
&& is_const<type_sans_ref>::value
&& !is_volatile<type_sans_ref>::value));
}; 


HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN
HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_BEGIN

template<typename From, typename To>
struct is_convertible_sfinae
{
private:
typedef char                          yes;
typedef struct { char two_chars[2]; } no;

static inline yes   test(To) { return yes(); }
static inline no    test(...) { return no(); } 
static inline typename remove_reference<From>::type& from() { typename remove_reference<From>::type* ptr = 0; return *ptr; }

public:
static const bool value = sizeof(test(from())) == sizeof(yes);
}; 


HYDRA_THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_END
HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END


template<typename From, typename To>
struct is_convertible_needs_simple_test
{
static const bool from_is_void      = is_void<From>::value;
static const bool to_is_void        = is_void<To>::value;
static const bool from_is_float     = is_floating_point<typename remove_reference<From>::type>::value;
static const bool to_is_int_or_cref = is_int_or_cref<To>::value;

static const bool value = (from_is_void || to_is_void || (from_is_float && to_is_int_or_cref));
}; 


template<typename From, typename To,
bool = is_convertible_needs_simple_test<From,To>::value>
struct is_convertible
{
static const bool value = (is_void<To>::value
|| (is_int_or_cref<To>::value
&& !is_void<From>::value));
}; 


template<typename From, typename To>
struct is_convertible<From, To, false>
{
static const bool value = (is_convertible_sfinae<typename
add_reference<From>::type, To>::value);
}; 


} 

template<typename From, typename To>
struct is_convertible
: public integral_constant<bool, tt_detail::is_convertible<From, To>::value>
{
}; 

#endif

template<typename T1, typename T2>
struct is_one_convertible_to_the_other
: public integral_constant<
bool,
is_convertible<T1,T2>::value || is_convertible<T2,T1>::value
>
{};



template <typename Condition1,               typename Condition2,              typename Condition3 = false_type,
typename Condition4  = false_type, typename Condition5 = false_type, typename Condition6 = false_type,
typename Condition7  = false_type, typename Condition8 = false_type, typename Condition9 = false_type,
typename Condition10 = false_type>
struct or_
: public integral_constant<
bool,
Condition1::value || Condition2::value || Condition3::value || Condition4::value || Condition5::value || Condition6::value || Condition7::value || Condition8::value || Condition9::value || Condition10::value
>
{
}; 

template <typename Condition1, typename Condition2, typename Condition3 = true_type>
struct and_
: public integral_constant<bool, Condition1::value && Condition2::value && Condition3::value>
{
}; 

template <typename Boolean>
struct not_
: public integral_constant<bool, !Boolean::value>
{
}; 

template<bool B, class T, class F>
struct conditional { typedef T type; };

template<class T, class F>
struct conditional<false, T, F> { typedef F type; };

template <bool, typename Then, typename Else>
struct eval_if
{
}; 

template<typename Then, typename Else>
struct eval_if<true, Then, Else>
{
typedef typename Then::type type;
}; 

template<typename Then, typename Else>
struct eval_if<false, Then, Else>
{
typedef typename Else::type type;
}; 

template<typename T>
struct identity_
{
typedef T type;
}; 

template<bool, typename T = void> struct enable_if {};
template<typename T>              struct enable_if<true, T> {typedef T type;};

template<bool, typename T> struct lazy_enable_if {};
template<typename T>       struct lazy_enable_if<true, T> {typedef typename T::type type;};

template<bool condition, typename T = void> struct disable_if : enable_if<!condition, T> {};
template<bool condition, typename T>        struct lazy_disable_if : lazy_enable_if<!condition, T> {};


template<typename T1, typename T2, typename T = void>
struct enable_if_convertible
: enable_if< is_convertible<T1,T2>::value, T >
{};


template<typename T1, typename T2, typename T = void>
struct disable_if_convertible
: disable_if< is_convertible<T1,T2>::value, T >
{};


template<typename T1, typename T2, typename Result = void>
struct enable_if_different
: enable_if<is_different<T1,T2>::value, Result>
{};


template<typename T>
struct is_numeric
: and_<
is_convertible<int,T>,
is_convertible<T,int>
>
{
}; 


template<typename> struct is_reference_to_const             : false_type {};
template<typename T> struct is_reference_to_const<const T&> : true_type {};



namespace tt_detail
{

template<typename T> struct make_unsigned_simple;

template<> struct make_unsigned_simple<char>                   { typedef unsigned char          type; };
template<> struct make_unsigned_simple<signed char>            { typedef unsigned char          type; };
template<> struct make_unsigned_simple<unsigned char>          { typedef unsigned char          type; };
template<> struct make_unsigned_simple<short>                  { typedef unsigned short         type; };
template<> struct make_unsigned_simple<unsigned short>         { typedef unsigned short         type; };
template<> struct make_unsigned_simple<int>                    { typedef unsigned int           type; };
template<> struct make_unsigned_simple<unsigned int>           { typedef unsigned int           type; };
template<> struct make_unsigned_simple<long int>               { typedef unsigned long int      type; };
template<> struct make_unsigned_simple<unsigned long int>      { typedef unsigned long int      type; };
template<> struct make_unsigned_simple<long long int>          { typedef unsigned long long int type; };
template<> struct make_unsigned_simple<unsigned long long int> { typedef unsigned long long int type; };

template<typename T>
struct make_unsigned_base
{
typedef typename remove_cv<T>::type remove_cv_t;

typedef typename make_unsigned_simple<remove_cv_t>::type unsigned_remove_cv_t;

typedef typename eval_if<
is_const<T>::value && is_volatile<T>::value,
add_cv<unsigned_remove_cv_t>,
eval_if<
is_const<T>::value,
add_const<unsigned_remove_cv_t>,
eval_if<
is_volatile<T>::value,
add_volatile<unsigned_remove_cv_t>,
identity_<unsigned_remove_cv_t>
>
>
>::type type;
};

} 

template<typename T>
struct make_unsigned
: tt_detail::make_unsigned_base<T>
{};

struct largest_available_float
{
#if defined(__CUDA_ARCH__)
#  if (__CUDA_ARCH__ < 130)
typedef float type;
#  else
typedef double type;
#  endif
#else
typedef double type;
#endif
};

template<typename T1, typename T2>
struct larger_type
: hydra_thrust::detail::eval_if<
(sizeof(T2) > sizeof(T1)),
hydra_thrust::detail::identity_<T2>,
hydra_thrust::detail::identity_<T1>
>
{};

#if HYDRA_THRUST_CPP_DIALECT >= 2011

using std::is_base_of;

#else

namespace is_base_of_ns
{

typedef char                          yes;
typedef struct { char two_chars[2]; } no;

template<typename Base, typename Derived>
struct host
{
operator Base*() const;
operator Derived*();
}; 

template<typename Base, typename Derived>
struct impl
{
template<typename T> static yes check(Derived *, T);
static no check(Base*, int);

static const bool value = sizeof(check(host<Base,Derived>(), int())) == sizeof(yes);
}; 

} 


template<typename Base, typename Derived>
struct is_base_of
: integral_constant<
bool,
is_base_of_ns::impl<Base,Derived>::value
>
{};

#endif

template<typename Base, typename Derived, typename Result = void>
struct enable_if_base_of
: enable_if<
is_base_of<Base,Derived>::value,
Result
>
{};


namespace is_assignable_ns
{

template<typename T1, typename T2>
class is_assignable
{
typedef char                      yes_type;
typedef struct { char array[2]; } no_type;

template<typename T> static typename add_reference<T>::type declval();

template<unsigned int> struct helper { typedef void * type; };

template<typename U1, typename U2> static yes_type test(typename helper<sizeof(declval<U1>() = declval<U2>())>::type);

template<typename,typename> static no_type test(...);

public:
static const bool value = sizeof(test<T1,T2>(0)) == 1;
}; 

} 


template<typename T1, typename T2>
struct is_assignable
: integral_constant<
bool,
is_assignable_ns::is_assignable<T1,T2>::value
>
{};


template<typename T>
struct is_copy_assignable
: is_assignable<
typename add_reference<T>::type,
typename add_reference<typename add_const<T>::type>::type
>
{};


template<typename T1, typename T2, typename Enable = void> struct promoted_numerical_type;

template<typename T1, typename T2> 
struct promoted_numerical_type<T1,T2,typename enable_if<and_
<typename is_floating_point<T1>::type,typename is_floating_point<T2>::type>
::value>::type>
{
typedef typename larger_type<T1,T2>::type type;
};

template<typename T1, typename T2> 
struct promoted_numerical_type<T1,T2,typename enable_if<and_
<typename is_integral<T1>::type,typename is_floating_point<T2>::type>
::value>::type>
{
typedef T2 type;
};

template<typename T1, typename T2>
struct promoted_numerical_type<T1,T2,typename enable_if<and_
<typename is_floating_point<T1>::type, typename is_integral<T2>::type>
::value>::type>
{
typedef T1 type;
};

template<typename T>
struct is_empty_helper : public T
{
};

struct is_empty_helper_base
{
};

template<typename T>
struct is_empty : integral_constant<bool,
sizeof(is_empty_helper_base) == sizeof(is_empty_helper<T>)
>
{
};

} 

using detail::integral_constant;
using detail::true_type;
using detail::false_type;

} 

#include <hydra/detail/external/hydra_thrust/detail/type_traits/has_trivial_assign.h>
