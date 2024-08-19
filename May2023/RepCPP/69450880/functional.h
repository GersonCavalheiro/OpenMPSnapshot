




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <functional>
#include <hydra/detail/external/hydra_thrust/detail/functional/placeholder.h>

namespace hydra_thrust
{



template<typename Operation> struct unary_traits;

template<typename Operation> struct binary_traits;




template<typename Argument,
typename Result>
struct unary_function
{

typedef Argument argument_type;


typedef Result   result_type;
}; 


template<typename Argument1,
typename Argument2,
typename Result>
struct binary_function
{

typedef Argument1 first_argument_type;


typedef Argument2 second_argument_type;


typedef Result    result_type;
}; 









template<typename T>
struct plus
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs + rhs;}
}; 


template<typename T>
struct minus
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs - rhs;}
}; 


template<typename T>
struct multiplies
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs * rhs;}
}; 


template<typename T>
struct divides
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs / rhs;}
}; 


template<typename T>
struct modulus
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs % rhs;}
}; 


template<typename T>
struct negate
{

typedef T argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &x) const {return -x;}
}; 


template<typename T>
struct square
{

typedef T argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &x) const {return x*x;}
}; 






template<typename T>
struct equal_to
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef bool result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs == rhs;}
}; 


template<typename T>
struct not_equal_to
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef bool result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs != rhs;}
}; 


template<typename T>
struct greater
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef bool result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs > rhs;}
}; 


template<typename T>
struct less
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef bool result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs < rhs;}
}; 


template<typename T>
struct greater_equal
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef bool result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs >= rhs;}
}; 


template<typename T>
struct less_equal
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef bool result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs <= rhs;}
}; 







template<typename T>
struct logical_and
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef bool result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs && rhs;}
}; 


template<typename T>
struct logical_or
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef bool result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs || rhs;}
}; 


template<typename T>
struct logical_not
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef bool result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ bool operator()(const T &x) const {return !x;}
}; 






template<typename T>
struct bit_and
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs & rhs;}
}; 


template<typename T>
struct bit_or
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs | rhs;}
}; 


template<typename T>
struct bit_xor
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs ^ rhs;}
}; 






template<typename T>
struct identity
{

typedef T argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ const T &operator()(const T &x) const {return x;}
}; 


template<typename T>
struct maximum
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs < rhs ? rhs : lhs;}
}; 


template<typename T>
struct minimum
{

typedef T first_argument_type;


typedef T second_argument_type;


typedef T result_type;


__hydra_thrust_exec_check_disable__
__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return lhs < rhs ? lhs : rhs;}
}; 


template<typename T1, typename T2>
struct project1st
{

typedef T1 first_argument_type;


typedef T2 second_argument_type;


typedef T1 result_type;


__host__ __device__ const T1 &operator()(const T1 &lhs, const T2 & ) const {return lhs;}
}; 


template<typename T1, typename T2>
struct project2nd
{

typedef T1 first_argument_type;


typedef T2 second_argument_type;


typedef T2 result_type;


__host__ __device__ const T2 &operator()(const T1 &, const T2 &rhs) const {return rhs;}
}; 








template<typename Predicate>
struct unary_negate 
: public hydra_thrust::unary_function<typename Predicate::argument_type, bool>
{

__host__ __device__
explicit unary_negate(Predicate p) : pred(p){}


__hydra_thrust_exec_check_disable__
__host__ __device__
bool operator()(const typename Predicate::argument_type& x) { return !pred(x); }


Predicate pred;

}; 


template<typename Predicate>
__host__ __device__
unary_negate<Predicate> not1(const Predicate &pred);


template<typename Predicate>
struct binary_negate
: public hydra_thrust::binary_function<typename Predicate::first_argument_type,
typename Predicate::second_argument_type,
bool>
{

__host__ __device__
explicit binary_negate(Predicate p) : pred(p){}


__hydra_thrust_exec_check_disable__
__host__ __device__
bool operator()(const typename Predicate::first_argument_type& x, const typename Predicate::second_argument_type& y)
{ 
return !pred(x,y); 
}


Predicate pred;

}; 


template<typename BinaryPredicate>
__host__ __device__
binary_negate<BinaryPredicate> not2(const BinaryPredicate &pred);








namespace placeholders
{



#ifdef __CUDA_ARCH__
static const __device__ hydra_thrust::detail::functional::placeholder<0>::type _1;
#else
static const hydra_thrust::detail::functional::placeholder<0>::type _1;
#endif



#ifdef __CUDA_ARCH__
static const __device__ hydra_thrust::detail::functional::placeholder<1>::type _2;
#else
static const hydra_thrust::detail::functional::placeholder<1>::type _2;
#endif



#ifdef __CUDA_ARCH__
static const __device__ hydra_thrust::detail::functional::placeholder<2>::type _3;
#else
static const hydra_thrust::detail::functional::placeholder<2>::type _3;
#endif



#ifdef __CUDA_ARCH__
static const __device__ hydra_thrust::detail::functional::placeholder<3>::type _4;
#else
static const hydra_thrust::detail::functional::placeholder<3>::type _4;
#endif



#ifdef __CUDA_ARCH__
static const __device__ hydra_thrust::detail::functional::placeholder<4>::type _5;
#else
static const hydra_thrust::detail::functional::placeholder<4>::type _5;
#endif



#ifdef __CUDA_ARCH__
static const __device__ hydra_thrust::detail::functional::placeholder<5>::type _6;
#else
static const hydra_thrust::detail::functional::placeholder<5>::type _6;
#endif



#ifdef __CUDA_ARCH__
static const __device__ hydra_thrust::detail::functional::placeholder<6>::type _7;
#else
static const hydra_thrust::detail::functional::placeholder<6>::type _7;
#endif



#ifdef __CUDA_ARCH__
static const __device__ hydra_thrust::detail::functional::placeholder<7>::type _8;
#else
static const hydra_thrust::detail::functional::placeholder<7>::type _8;
#endif



#ifdef __CUDA_ARCH__
static const __device__ hydra_thrust::detail::functional::placeholder<8>::type _9;
#else
static const hydra_thrust::detail::functional::placeholder<8>::type _9;
#endif



#ifdef __CUDA_ARCH__
static const __device__ hydra_thrust::detail::functional::placeholder<9>::type _10;
#else
static const hydra_thrust::detail::functional::placeholder<9>::type _10;
#endif


} 





} 

#include <hydra/detail/external/hydra_thrust/detail/functional.inl>
#include <hydra/detail/external/hydra_thrust/detail/functional/operators.h>

