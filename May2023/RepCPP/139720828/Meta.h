
#ifndef EIGEN_META_H
#define EIGEN_META_H

#if defined(__CUDA_ARCH__)
#include <cfloat>
#include <math_constants.h>
#endif

#if EIGEN_COMP_ICC>=1600 &&  __cplusplus >= 201103L
#include <cstdint>
#endif

namespace Eigen {

typedef EIGEN_DEFAULT_DENSE_INDEX_TYPE DenseIndex;



typedef EIGEN_DEFAULT_DENSE_INDEX_TYPE Index;

namespace internal {



#if EIGEN_COMP_ICC>=1600 &&  __cplusplus >= 201103L
typedef std::intptr_t  IntPtr;
typedef std::uintptr_t UIntPtr;
#else
typedef std::ptrdiff_t IntPtr;
typedef std::size_t UIntPtr;
#endif

struct true_type {  enum { value = 1 }; };
struct false_type { enum { value = 0 }; };

template<bool Condition, typename Then, typename Else>
struct conditional { typedef Then type; };

template<typename Then, typename Else>
struct conditional <false, Then, Else> { typedef Else type; };

template<typename T, typename U> struct is_same { enum { value = 0 }; };
template<typename T> struct is_same<T,T> { enum { value = 1 }; };

template<typename T> struct remove_reference { typedef T type; };
template<typename T> struct remove_reference<T&> { typedef T type; };

template<typename T> struct remove_pointer { typedef T type; };
template<typename T> struct remove_pointer<T*> { typedef T type; };
template<typename T> struct remove_pointer<T*const> { typedef T type; };

template <class T> struct remove_const { typedef T type; };
template <class T> struct remove_const<const T> { typedef T type; };
template <class T> struct remove_const<const T[]> { typedef T type[]; };
template <class T, unsigned int Size> struct remove_const<const T[Size]> { typedef T type[Size]; };

template<typename T> struct remove_all { typedef T type; };
template<typename T> struct remove_all<const T>   { typedef typename remove_all<T>::type type; };
template<typename T> struct remove_all<T const&>  { typedef typename remove_all<T>::type type; };
template<typename T> struct remove_all<T&>        { typedef typename remove_all<T>::type type; };
template<typename T> struct remove_all<T const*>  { typedef typename remove_all<T>::type type; };
template<typename T> struct remove_all<T*>        { typedef typename remove_all<T>::type type; };

template<typename T> struct is_arithmetic      { enum { value = false }; };
template<> struct is_arithmetic<float>         { enum { value = true }; };
template<> struct is_arithmetic<double>        { enum { value = true }; };
template<> struct is_arithmetic<long double>   { enum { value = true }; };
template<> struct is_arithmetic<bool>          { enum { value = true }; };
template<> struct is_arithmetic<char>          { enum { value = true }; };
template<> struct is_arithmetic<signed char>   { enum { value = true }; };
template<> struct is_arithmetic<unsigned char> { enum { value = true }; };
template<> struct is_arithmetic<signed short>  { enum { value = true }; };
template<> struct is_arithmetic<unsigned short>{ enum { value = true }; };
template<> struct is_arithmetic<signed int>    { enum { value = true }; };
template<> struct is_arithmetic<unsigned int>  { enum { value = true }; };
template<> struct is_arithmetic<signed long>   { enum { value = true }; };
template<> struct is_arithmetic<unsigned long> { enum { value = true }; };

template<typename T> struct is_integral        { enum { value = false }; };
template<> struct is_integral<bool>            { enum { value = true }; };
template<> struct is_integral<char>            { enum { value = true }; };
template<> struct is_integral<signed char>     { enum { value = true }; };
template<> struct is_integral<unsigned char>   { enum { value = true }; };
template<> struct is_integral<signed short>    { enum { value = true }; };
template<> struct is_integral<unsigned short>  { enum { value = true }; };
template<> struct is_integral<signed int>      { enum { value = true }; };
template<> struct is_integral<unsigned int>    { enum { value = true }; };
template<> struct is_integral<signed long>     { enum { value = true }; };
template<> struct is_integral<unsigned long>   { enum { value = true }; };

template <typename T> struct add_const { typedef const T type; };
template <typename T> struct add_const<T&> { typedef T& type; };

template <typename T> struct is_const { enum { value = 0 }; };
template <typename T> struct is_const<T const> { enum { value = 1 }; };

template<typename T> struct add_const_on_value_type            { typedef const T type;  };
template<typename T> struct add_const_on_value_type<T&>        { typedef T const& type; };
template<typename T> struct add_const_on_value_type<T*>        { typedef T const* type; };
template<typename T> struct add_const_on_value_type<T* const>  { typedef T const* const type; };
template<typename T> struct add_const_on_value_type<T const* const>  { typedef T const* const type; };


template<typename From, typename To>
struct is_convertible_impl
{
private:
struct any_conversion
{
template <typename T> any_conversion(const volatile T&);
template <typename T> any_conversion(T&);
};
struct yes {int a[1];};
struct no  {int a[2];};

static yes test(const To&, int);
static no  test(any_conversion, ...);

public:
static From ms_from;
#ifdef __INTEL_COMPILER
#pragma warning push
#pragma warning ( disable : 2259 )
#endif
enum { value = sizeof(test(ms_from, 0))==sizeof(yes) };
#ifdef __INTEL_COMPILER
#pragma warning pop
#endif
};

template<typename From, typename To>
struct is_convertible
{
enum { value = is_convertible_impl<typename remove_all<From>::type,
typename remove_all<To  >::type>::value };
};


template<bool Condition, typename T=void> struct enable_if;

template<typename T> struct enable_if<true,T>
{ typedef T type; };

#if defined(__CUDA_ARCH__)
#if !defined(__FLT_EPSILON__)
#define __FLT_EPSILON__ FLT_EPSILON
#define __DBL_EPSILON__ DBL_EPSILON
#endif

namespace device {

template<typename T> struct numeric_limits
{
EIGEN_DEVICE_FUNC
static T epsilon() { return 0; }
static T (max)() { assert(false && "Highest not supported for this type"); }
static T (min)() { assert(false && "Lowest not supported for this type"); }
static T infinity() { assert(false && "Infinity not supported for this type"); }
static T quiet_NaN() { assert(false && "quiet_NaN not supported for this type"); }
};
template<> struct numeric_limits<float>
{
EIGEN_DEVICE_FUNC
static float epsilon() { return __FLT_EPSILON__; }
EIGEN_DEVICE_FUNC
static float (max)() { return CUDART_MAX_NORMAL_F; }
EIGEN_DEVICE_FUNC
static float (min)() { return FLT_MIN; }
EIGEN_DEVICE_FUNC
static float infinity() { return CUDART_INF_F; }
EIGEN_DEVICE_FUNC
static float quiet_NaN() { return CUDART_NAN_F; }
};
template<> struct numeric_limits<double>
{
EIGEN_DEVICE_FUNC
static double epsilon() { return __DBL_EPSILON__; }
EIGEN_DEVICE_FUNC
static double (max)() { return DBL_MAX; }
EIGEN_DEVICE_FUNC
static double (min)() { return DBL_MIN; }
EIGEN_DEVICE_FUNC
static double infinity() { return CUDART_INF; }
EIGEN_DEVICE_FUNC
static double quiet_NaN() { return CUDART_NAN; }
};
template<> struct numeric_limits<int>
{
EIGEN_DEVICE_FUNC
static int epsilon() { return 0; }
EIGEN_DEVICE_FUNC
static int (max)() { return INT_MAX; }
EIGEN_DEVICE_FUNC
static int (min)() { return INT_MIN; }
};
template<> struct numeric_limits<unsigned int>
{
EIGEN_DEVICE_FUNC
static unsigned int epsilon() { return 0; }
EIGEN_DEVICE_FUNC
static unsigned int (max)() { return UINT_MAX; }
EIGEN_DEVICE_FUNC
static unsigned int (min)() { return 0; }
};
template<> struct numeric_limits<long>
{
EIGEN_DEVICE_FUNC
static long epsilon() { return 0; }
EIGEN_DEVICE_FUNC
static long (max)() { return LONG_MAX; }
EIGEN_DEVICE_FUNC
static long (min)() { return LONG_MIN; }
};
template<> struct numeric_limits<unsigned long>
{
EIGEN_DEVICE_FUNC
static unsigned long epsilon() { return 0; }
EIGEN_DEVICE_FUNC
static unsigned long (max)() { return ULONG_MAX; }
EIGEN_DEVICE_FUNC
static unsigned long (min)() { return 0; }
};
template<> struct numeric_limits<long long>
{
EIGEN_DEVICE_FUNC
static long long epsilon() { return 0; }
EIGEN_DEVICE_FUNC
static long long (max)() { return LLONG_MAX; }
EIGEN_DEVICE_FUNC
static long long (min)() { return LLONG_MIN; }
};
template<> struct numeric_limits<unsigned long long>
{
EIGEN_DEVICE_FUNC
static unsigned long long epsilon() { return 0; }
EIGEN_DEVICE_FUNC
static unsigned long long (max)() { return ULLONG_MAX; }
EIGEN_DEVICE_FUNC
static unsigned long long (min)() { return 0; }
};

}

#endif


class noncopyable
{
EIGEN_DEVICE_FUNC noncopyable(const noncopyable&);
EIGEN_DEVICE_FUNC const noncopyable& operator=(const noncopyable&);
protected:
EIGEN_DEVICE_FUNC noncopyable() {}
EIGEN_DEVICE_FUNC ~noncopyable() {}
};


#if EIGEN_HAS_STD_RESULT_OF
template<typename T> struct result_of {
typedef typename std::result_of<T>::type type1;
typedef typename remove_all<type1>::type type;
};
#else
template<typename T> struct result_of { };

struct has_none {int a[1];};
struct has_std_result_type {int a[2];};
struct has_tr1_result {int a[3];};

template<typename Func, typename ArgType, int SizeOf=sizeof(has_none)>
struct unary_result_of_select {typedef typename internal::remove_all<ArgType>::type type;};

template<typename Func, typename ArgType>
struct unary_result_of_select<Func, ArgType, sizeof(has_std_result_type)> {typedef typename Func::result_type type;};

template<typename Func, typename ArgType>
struct unary_result_of_select<Func, ArgType, sizeof(has_tr1_result)> {typedef typename Func::template result<Func(ArgType)>::type type;};

template<typename Func, typename ArgType>
struct result_of<Func(ArgType)> {
template<typename T>
static has_std_result_type    testFunctor(T const *, typename T::result_type const * = 0);
template<typename T>
static has_tr1_result         testFunctor(T const *, typename T::template result<T(ArgType)>::type const * = 0);
static has_none               testFunctor(...);

enum {FunctorType = sizeof(testFunctor(static_cast<Func*>(0)))};
typedef typename unary_result_of_select<Func, ArgType, FunctorType>::type type;
};

template<typename Func, typename ArgType0, typename ArgType1, int SizeOf=sizeof(has_none)>
struct binary_result_of_select {typedef typename internal::remove_all<ArgType0>::type type;};

template<typename Func, typename ArgType0, typename ArgType1>
struct binary_result_of_select<Func, ArgType0, ArgType1, sizeof(has_std_result_type)>
{typedef typename Func::result_type type;};

template<typename Func, typename ArgType0, typename ArgType1>
struct binary_result_of_select<Func, ArgType0, ArgType1, sizeof(has_tr1_result)>
{typedef typename Func::template result<Func(ArgType0,ArgType1)>::type type;};

template<typename Func, typename ArgType0, typename ArgType1>
struct result_of<Func(ArgType0,ArgType1)> {
template<typename T>
static has_std_result_type    testFunctor(T const *, typename T::result_type const * = 0);
template<typename T>
static has_tr1_result         testFunctor(T const *, typename T::template result<T(ArgType0,ArgType1)>::type const * = 0);
static has_none               testFunctor(...);

enum {FunctorType = sizeof(testFunctor(static_cast<Func*>(0)))};
typedef typename binary_result_of_select<Func, ArgType0, ArgType1, FunctorType>::type type;
};

template<typename Func, typename ArgType0, typename ArgType1, typename ArgType2, int SizeOf=sizeof(has_none)>
struct ternary_result_of_select {typedef typename internal::remove_all<ArgType0>::type type;};

template<typename Func, typename ArgType0, typename ArgType1, typename ArgType2>
struct ternary_result_of_select<Func, ArgType0, ArgType1, ArgType2, sizeof(has_std_result_type)>
{typedef typename Func::result_type type;};

template<typename Func, typename ArgType0, typename ArgType1, typename ArgType2>
struct ternary_result_of_select<Func, ArgType0, ArgType1, ArgType2, sizeof(has_tr1_result)>
{typedef typename Func::template result<Func(ArgType0,ArgType1,ArgType2)>::type type;};

template<typename Func, typename ArgType0, typename ArgType1, typename ArgType2>
struct result_of<Func(ArgType0,ArgType1,ArgType2)> {
template<typename T>
static has_std_result_type    testFunctor(T const *, typename T::result_type const * = 0);
template<typename T>
static has_tr1_result         testFunctor(T const *, typename T::template result<T(ArgType0,ArgType1,ArgType2)>::type const * = 0);
static has_none               testFunctor(...);

enum {FunctorType = sizeof(testFunctor(static_cast<Func*>(0)))};
typedef typename ternary_result_of_select<Func, ArgType0, ArgType1, ArgType2, FunctorType>::type type;
};
#endif

struct meta_yes { char a[1]; };
struct meta_no  { char a[2]; };

template <typename T>
struct has_ReturnType
{
template <typename C> static meta_yes testFunctor(typename C::ReturnType const *);
template <typename C> static meta_no testFunctor(...);

enum { value = sizeof(testFunctor<T>(0)) == sizeof(meta_yes) };
};

template<typename T> const T* return_ptr();

template <typename T, typename IndexType=Index>
struct has_nullary_operator
{
template <typename C> static meta_yes testFunctor(C const *,typename enable_if<(sizeof(return_ptr<C>()->operator()())>0)>::type * = 0);
static meta_no testFunctor(...);

enum { value = sizeof(testFunctor(static_cast<T*>(0))) == sizeof(meta_yes) };
};

template <typename T, typename IndexType=Index>
struct has_unary_operator
{
template <typename C> static meta_yes testFunctor(C const *,typename enable_if<(sizeof(return_ptr<C>()->operator()(IndexType(0)))>0)>::type * = 0);
static meta_no testFunctor(...);

enum { value = sizeof(testFunctor(static_cast<T*>(0))) == sizeof(meta_yes) };
};

template <typename T, typename IndexType=Index>
struct has_binary_operator
{
template <typename C> static meta_yes testFunctor(C const *,typename enable_if<(sizeof(return_ptr<C>()->operator()(IndexType(0),IndexType(0)))>0)>::type * = 0);
static meta_no testFunctor(...);

enum { value = sizeof(testFunctor(static_cast<T*>(0))) == sizeof(meta_yes) };
};


template<int Y,
int InfX = 0,
int SupX = ((Y==1) ? 1 : Y/2),
bool Done = ((SupX-InfX)<=1 ? true : ((SupX*SupX <= Y) && ((SupX+1)*(SupX+1) > Y))) >
class meta_sqrt
{
enum {
MidX = (InfX+SupX)/2,
TakeInf = MidX*MidX > Y ? 1 : 0,
NewInf = int(TakeInf) ? InfX : int(MidX),
NewSup = int(TakeInf) ? int(MidX) : SupX
};
public:
enum { ret = meta_sqrt<Y,NewInf,NewSup>::ret };
};

template<int Y, int InfX, int SupX>
class meta_sqrt<Y, InfX, SupX, true> { public:  enum { ret = (SupX*SupX <= Y) ? SupX : InfX }; };



template<int A, int B, int K=1, bool Done = ((A*K)%B)==0>
struct meta_least_common_multiple
{
enum { ret = meta_least_common_multiple<A,B,K+1>::ret };
};
template<int A, int B, int K>
struct meta_least_common_multiple<A,B,K,true>
{
enum { ret = A*K };
};


template<typename T, typename U> struct scalar_product_traits
{
enum { Defined = 0 };
};


} 

namespace numext {

#if defined(__CUDA_ARCH__)
template<typename T> EIGEN_DEVICE_FUNC   void swap(T &a, T &b) { T tmp = b; b = a; a = tmp; }
#else
template<typename T> EIGEN_STRONG_INLINE void swap(T &a, T &b) { std::swap(a,b); }
#endif

#if defined(__CUDA_ARCH__)
using internal::device::numeric_limits;
#else
using std::numeric_limits;
#endif

template<typename T>
T div_ceil(const T &a, const T &b)
{
return (a+b-1) / b;
}

} 

} 

#endif 
